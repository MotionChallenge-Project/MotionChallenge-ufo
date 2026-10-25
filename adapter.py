"""UFO FB-CPR-Aux: released backward encoder and native 50 Hz deployment."""
from motion_challenge.adapters.onnx import cpu_session
import numpy as np

from motion_challenge.adapters.onnx import ONNXController, vector
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


def rotate(q, v, inverse=False):
    """Unit-quaternion rotation in xyzw order, preserving input precision."""
    xyz, w = q[..., :3], q[..., 3:4]
    cross = 2 * w * np.cross(xyz, v)
    return v * (2*w*w-1) + (-cross if inverse else cross) + 2*xyz*np.sum(xyz*v, axis=-1, keepdims=True)


def multiply(a, b):
    xyz = a[..., 3:4]*b[..., :3] + b[..., 3:4]*a[..., :3] + np.cross(a[..., :3], b[..., :3])
    w = a[..., 3:4]*b[..., 3:4] - np.sum(a[..., :3]*b[..., :3], axis=-1, keepdims=True)
    return np.concatenate((xyz, w), axis=-1)


class UFO(ONNXController):
    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        item = runtime['artifacts']['encoder']
        path = root / item['path']
        if sha256_file(path) != item['sha256']:
            raise ValueError('encoder checksum mismatch')
        self.encoder = cpu_session(path, item['sha256'])
        self.lower = vector(self.cfg['joint_pos_lower_limit'], self.names)
        self.upper = vector(self.cfg['joint_pos_upper_limit'], self.names)
        self.max_delta = vector(self.cfg['joint_velocity_limit'], self.names)*runtime['control_period_s']*self.cfg['q_target_slew_safety_factor']
        assert self.session.get_inputs()[0].shape[-1] == 721
        assert [(n.name, n.shape[-1]) for n in self.encoder.get_inputs()] == [('state', 64), ('privileged_state', 463)]

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.history = {key: np.zeros((4, size)) for key, size in (
            ('prev_actions', 29), ('base_ang_vel', 3), ('dof_pos_minus_default', 29),
            ('dof_vel', 29), ('projected_gravity', 3))}
        self.last_target = None
        self.context_cache = {}
        self.context_norm = None
        self.last_context = None

    def reference_observations(self, frame):
        bodies = {b.name: b for b in frame.bodies}
        ordered = [bodies[n] for n in self.cfg['reference_body_names']]
        pos = np.asarray([b.position_world_m for b in ordered], np.float32)
        quat = np.asarray([b.orientation_xyzw for b in ordered], np.float32)
        vel = np.asarray([b.linear_velocity_world_m_s for b in ordered], np.float32)
        omega = np.asarray([b.angular_velocity_world_rad_s for b in ordered], np.float32)
        # The released encoder includes one virtual head, rigidly attached to torso.
        extension = self.cfg['reference_motion_extensions'][0]
        parent = self.cfg['reference_body_names'].index(extension['parent_name'])
        offset = rotate(quat[parent], np.asarray(extension['pos'], np.float32))
        pos = np.vstack((pos, pos[parent]+offset))
        quat = np.vstack((quat, quat[parent]))
        vel = np.vstack((vel, vel[parent]+np.cross(omega[parent], offset)))
        omega = np.vstack((omega, omega[parent]))
        forward = rotate(quat[0], np.array([1, 0, 0], np.float32))
        angle = -np.arctan2(forward[1], forward[0])
        heading = np.array([0, 0, np.sin(angle/np.float32(2)), np.cos(angle/np.float32(2))], np.float32)
        heading /= np.linalg.norm(heading)
        local_quat = multiply(heading, quat)
        local_rotation = np.concatenate((rotate(local_quat, np.array([1, 0, 0], np.float32)),
                                         rotate(local_quat, np.array([0, 0, 1], np.float32))), axis=-1)
        privileged = np.concatenate((pos[0, 2:3], rotate(heading, pos-pos[0])[1:].ravel(),
            local_rotation.ravel(), rotate(heading, vel).ravel(), rotate(heading, omega).ravel()))
        root = np.asarray(frame.root_orientation_xyzw, np.float32)
        state = np.concatenate((np.asarray(frame.joint_position_rad, np.float32)[self.order]-self.default.astype(np.float32),
            np.asarray(frame.joint_velocity_rad_s, np.float32)[self.order],
            rotate(root, np.array([0, 0, -1], np.float32), inverse=True),
            rotate(root, np.asarray(frame.root_angular_velocity_world_rad_s, np.float32), inverse=True)))
        return {'state': state[None], 'privileged_state': privileged[None]}

    def context(self, reference):
        if len(reference) != 4:
            raise ValueError('UFO requires three future targets plus an endpoint probe')
        # Native exports exclude the exact endpoint. The fourth query only detects
        # endpoint clamping; its pose never enters the three-target latent window.
        ends = [f.source_time_s for f in reference if f.requested_time_s > f.source_time_s+1e-9]
        end = min(ends) if ends else float('inf')
        frames = [f for f in reference[:3] if f.requested_time_s < end-1e-9]
        window, cache = [], {}
        for frame in frames:
            key = round(frame.source_time_s, 10)
            z = self.context_cache.get(key)
            if z is None:
                z = self.encoder.run(['z'], self.reference_observations(frame))[0][0]
            cache[key] = z
            window.append(z)
        if window:
            self.last_context = window[-1]
            self.context_cache = cache
        elif self.last_context is not None:
            # Finish the canonical clip without the deployment viewer's wraparound.
            window = [self.last_context]
        else:
            window = [self.encoder.run(['z'], self.reference_observations(reference[0]))[0][0]]
            self.last_context = window[0]
        if self.context_norm is None:
            self.context_norm = float(np.linalg.norm(window[0]))
        discounts = self.cfg['gamma']**np.arange(len(window))
        discounts /= np.sum(discounts)
        z = np.sum(np.asarray(window)*discounts[:, None], axis=0)
        return z/(np.linalg.norm(z)+1e-8)*self.context_norm

    def observations(self, state, reference):
        q = np.asarray(state.joint_position_rad)[self.order]
        root = np.asarray(state.root_orientation_xyzw)
        values = {'dof_pos_minus_default': q-self.default,
            'dof_vel': np.asarray(state.joint_velocity_rad_s)[self.order],
            'projected_gravity': rotate(root, np.array([0, 0, -1]), inverse=True),
            'base_ang_vel': rotate(root, np.asarray(state.root_angular_velocity_world_rad_s), inverse=True),
            'prev_actions': self.last}
        obs = np.concatenate([self.history[k.removesuffix('_history')].ravel() if k.endswith('_history')
            else values[k] for k in self.cfg['observation_order']]).astype(np.float32)
        for key, history in self.history.items():
            history[1:] = history[:-1].copy()
            history[0] = values[key]
        return {'actor_obs': np.concatenate((obs, self.context(reference))).astype(np.float32)[None]}

    def act(self, state, reference):
        raw = self.infer(self.observations(state, reference))['action'].reshape(29)
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite action')
        self.last = np.clip(raw, -1, 1)*self.cfg['action_rescale']
        target = np.clip(self.default+self.scale*self.last, self.lower, self.upper)
        baseline = np.asarray(state.joint_position_rad)[self.order] if self.last_target is None else self.last_target
        target = np.clip(baseline+np.clip(target-baseline, -self.max_delta, self.max_delta), self.lower, self.upper)
        self.last_target = target.copy()
        def ordered(x):
            return tuple(float(v) for v in x[self.inverse])
        return ControlCommand(ordered(target), (0.,)*29, ordered(self.kp), ordered(self.kd), (0.,)*29)

    def close(self):
        self.encoder = None
        super().close()


def create(root, runtime):
    return UFO(root, runtime)
