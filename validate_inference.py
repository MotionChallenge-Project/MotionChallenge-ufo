"""Compare the adapter with hash-pinned author code, without hardware imports.

uv run --with torch validate_inference.py --source /path/to/source-bundle
  --weights /path/to/FBcprAuxModel.onnx --encoder /path/to/backward_encoder.onnx
  --output /outside/git/native-parity.json
"""
import argparse
import ast
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from adapter import UFO
from motion_challenge.physics import World, forward_kinematics
from motion_challenge.sdk import ReferenceFrame


def load_functions(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+nodes, []), str(path), 'exec'), scope)


def main():
    parser = argparse.ArgumentParser()
    for key in ('source', 'weights', 'encoder', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'native-sources.json').read_text())
    for folder, files in pins['files'].items():
        for name, digest in files.items():
            assert hashlib.sha256((args.source/folder/name).read_bytes()).hexdigest() == digest, name
    deploy, training = args.source/'ufo-release-source', args.source/'ufo-training-source'
    native = {'np': np, 'torch': torch, 'OrderedDict': OrderedDict, 'time': time,
              'logger': NS(info=lambda *_: None, warning=lambda *_: None)}
    load_functions(training/'ufo/math.py', ('normalize', 'quat_unit', 'quat_rotate_inverse',
        'quat_from_angle_axis', 'my_quat_rotate', 'quat_to_tan_norm', 'calc_heading',
        'calc_heading_quat_inv', 'quat_mul'), native)
    load_functions(training/'ufo/envs/observations.py', ('compute_humanoid_observations_max', 'scale_observation'), native)
    load_functions(training/'ufo/motion/reference.py', ('reference_base_ang_vel', '_motion_times', '_motion_observations'), native)
    load_functions(deploy/'utils/math.py', ('quat_rotate_inverse_numpy',), native)
    load_functions(deploy/'rl_policy/observations.py', ('Observation', 'ObsGroup'), native)
    tree = ast.parse((deploy/'rl_policy/ufo_policy.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UFODeployPolicy')
    methods = {'prepare_obs_for_rl', '_apply_q_target_slew_limit', '_rl_step_scheduled'}
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in cls.body} == methods
    exec(compile(ast.Module([cls], []), 'native-deploy', 'exec'), native)
    runtime = json.loads((root/'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    runtime['artifacts']['encoder']['path'] = str(args.encoder.resolve())
    policy = UFO(root, runtime)
    metadata = json.loads((args.source/'ufo-artifacts/exported/FBcprAuxModel.meta.json').read_text())
    assert hashlib.sha256((args.source/'ufo-artifacts/exported/FBcprAuxModel.meta.json').read_bytes()).hexdigest() == '41bef75f8644ea8127563023a41d1f1e2c9e3b03fa27485178bb8f7f57b1d971'
    robot = yaml.safe_load((deploy/'config/robot/g1.yaml').read_text())
    task = yaml.safe_load((deploy/'config/tasks/tracking.yaml').read_text())
    for key in ('joint_pos_lower_limit', 'joint_pos_upper_limit', 'joint_velocity_limit'):
        assert policy.cfg[key] == robot[key]
    assert policy.cfg['gamma'] == task['gamma'] == .8
    assert policy.cfg['window_size'] == task['window_size'] == 3
    assert policy.cfg['action_rescale'] == metadata['deployment']['action_rescale'] == 5.
    assert runtime['control_period_s'] == .02
    assert runtime['reference_offsets_s'] == [.02, .04, .06, .08]
    assert policy.names == metadata['control_joint_names']
    assert policy.cfg['observation_order'] == metadata['observation_order']
    for field, attr in [('default_joint_pos', 'default'), ('joint_kp', 'kp'), ('joint_kd', 'kd'), ('action_scale', 'scale')]:
        np.testing.assert_array_equal(getattr(policy, attr), [metadata['deployment'][field][n] for n in policy.names])
    world = World(root/'motion-challenge/robots/g1-29dof-cpu-v0.json')
    rng = np.random.default_rng(241024)
    shuffle = rng.permutation(29)
    names = tuple(np.asarray(world.joint_names)[shuffle])
    canonical = [names.index(n) for n in world.joint_names]
    policy_to_world = [world.joint_names.index(n) for n in policy.names]
    errors = {k: 0. for k in ('reference_state', 'reference_privileged', 'latent', 'actor_observation',
        'action', 'motor_target', 'history', 'reset_history', 'kp', 'kd', 'forced_clip_slew')}
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    def reorder(frame):
        return replace(frame, joint_position_rad=tuple(np.asarray(frame.joint_position_rad)[shuffle]),
                       joint_velocity_rad_s=tuple(np.asarray(frame.joint_velocity_rad_s)[shuffle]))
    def native_reference(frame):
        lookup = {b.name: b for b in frame.bodies}
        bodies = [lookup[n] for n in metadata['reference_body_names']]
        def tensor(attr):
            return torch.tensor([getattr(b, attr) for b in bodies], dtype=torch.float32)
        p, q = tensor('position_world_m'), tensor('orientation_xyzw')
        v, w = tensor('linear_velocity_world_m_s'), tensor('angular_velocity_world_rad_s')
        parent = metadata['reference_body_names'].index('torso_link')
        offset = native['my_quat_rotate'](q[parent:parent+1], torch.tensor([[0., 0., .35]]))
        p = torch.cat((p, p[parent:parent+1]+offset))
        v = torch.cat((v, v[parent:parent+1]+torch.cross(w[parent:parent+1], offset, dim=-1)))
        w = torch.cat((w, w[parent:parent+1]))
        q = torch.cat((q, q[parent:parent+1]))
        motion = {'rg_pos_t': p[None], 'rg_rot_t': q[None], 'body_vel_t': v[None], 'body_ang_vel_t': w[None],
            'dof_vel': torch.tensor(np.asarray(frame.joint_velocity_rad_s)[policy_to_world], dtype=torch.float32)[None]}
        env = NS(gravity_vec=torch.tensor([[0., 0., -1.]]), config=NS(obs=NS(obs_scales={}, noise_scales={})))
        relative = torch.tensor(np.asarray(frame.joint_position_rad)[policy_to_world], dtype=torch.float32)-torch.tensor(policy.default, dtype=torch.float32)
        state, privileged = native['_motion_observations'](env, motion, relative[None], root_height_obs=True, use_obs_filter=True)
        return {'state': state.numpy(), 'privileged_state': privileged.numpy()}
    case_count, endpoint_count = 0, 0
    for reset, duration in enumerate((.8, .795, .82, .815)):
        frames = []
        for index in range(43):
            rot = R.from_euler('xyz', [.1, np.pi/2-1e-5, -.5]) if index % 10 == 0 else R.random(random_state=rng)
            world.data.qpos[:] = np.r_[rng.normal(0, .4, 2), .8, rot.as_quat()[[3, 0, 1, 2]], rng.normal(0, .3, 29)]
            world.data.qvel[:] = rng.normal(0, .5, 35)
            forward_kinematics(world.model, world.data)
            s = world.state()
            frames.append(ReferenceFrame(index*.02, index*.02, s.joint_position_rad, s.root_position_world_m,
                s.root_orientation_xyzw, s.joint_velocity_rad_s, s.root_linear_velocity_world_m_s,
                s.root_angular_velocity_world_rad_s, s.bodies))
        s = replace(world.state(), joint_names=names,
            joint_position_rad=tuple(np.asarray(world.state().joint_position_rad)[shuffle]),
            joint_velocity_rad_s=tuple(np.asarray(world.state().joint_velocity_rad_s)[shuffle]))
        policy.reset(s, (), reset)
        compare('reset_history', np.concatenate([h.ravel() for h in policy.history.values()]), 0.)
        times = native['_motion_times'](NS(_motion_lib=NS(_motion_lengths=torch.tensor([duration])), dt=.02, device='cpu'), 0)
        contexts = []
        for index in range(1, len(times)):
            expected = native_reference(frames[index])
            actual = policy.reference_observations(reorder(frames[index]))
            compare('reference_state', actual['state'], expected['state'])
            compare('reference_privileged', actual['privileged_state'], expected['privileged_state'])
            expected_z = policy.encoder.run(['z'], expected)[0][0]
            compare('latent', policy.encoder.run(['z'], actual)[0][0], expected_z)
            contexts.append(expected_z)
        env = native['UFODeployPolicy']()
        env.state_processor = NS(num_dof=29, _prepare_low_state=lambda: True)
        env.num_dofs = env.num_actions = 29
        env.last_action = np.zeros(29)
        env.last_cmd_q = None
        env.controlled_joint_indices = np.arange(29)
        env.action_rescale = policy.cfg['action_rescale']
        env.action_scale, env.default_dof_angles = policy.scale, policy.default
        env.joint_pos_lower_limit, env.joint_pos_upper_limit = policy.lower, policy.upper
        env.joint_velocity_limit = np.array([policy.cfg['joint_velocity_limit'][n] for n in policy.names])
        env.q_target_slew_safety_factor = .5
        env.rl_dt = .02
        env.use_joystick = env.pico_control_enabled = env.stop_latched = env.hold_init_pose = False
        env.policy_armed = True
        env.start_motion = False
        env.task_type = 'tracking'
        env.gamma, env.window_size = .8, 3
        env.ctx = np.asarray(contexts)
        env.t_end = len(contexts)
        options = metadata['deployment']['observation']['policy']
        env.observations = {'policy': native['ObsGroup']('policy', {k: options[k] for k in metadata['observation_order']}, env)}
        capture = {}
        env.command_sender = NS(send_command=lambda q, qd, tau: capture.update(target=q.copy()))
        def native_infer(obs):
            capture['obs'] = obs.copy()
            return policy.session.run(['action'], {'actor_obs': obs})[0]
        env.policy = native_infer
        def fail(reason):
            raise AssertionError(reason)
        env.enter_stop_latch = fail
        for step in range(40):
            world.data.qpos[:] = world._qpos(frames[step])
            world.data.qpos[7:] += rng.normal(0, .2, 29)
            world.data.qvel[:] = rng.normal(0, .7, 35)
            forward_kinematics(world.model, world.data)
            state = world.state()
            env.state_processor.joint_pos = np.asarray(state.joint_position_rad)[policy_to_world]
            env.state_processor.joint_vel = np.asarray(state.joint_velocity_rad_s)[policy_to_world]
            env.state_processor.root_quat_b = np.asarray(state.root_orientation_xyzw)[[3, 0, 1, 2]]
            env.state_processor.root_ang_vel_b = native['quat_rotate_inverse_numpy'](env.state_processor.root_quat_b[None],
                np.asarray(state.root_angular_velocity_world_rad_s)[None])[0]
            state = replace(state, joint_names=names, time_s=step*.02,
                joint_position_rad=tuple(np.asarray(state.joint_position_rad)[shuffle]),
                joint_velocity_rad_s=tuple(np.asarray(state.joint_velocity_rad_s)[shuffle]))
            refs = tuple(reorder(replace(frames[min(step+i, int(duration/.02))],
                requested_time_s=(step+i)*.02, source_time_s=min((step+i)*.02, duration))) for i in range(1, 5))
            # The author window shrinks at EOF. Hold the final context only after
            # the exported sequence ends, to finish the benchmark's exact endpoint.
            env.t = env.t_stop = min(step, len(contexts)-1)
            endpoint_count += int(step+3 >= len(contexts))
            env._rl_step_scheduled()
            infer = policy.infer
            def capture_infer(feed):
                compare('actor_observation', feed['actor_obs'], capture['obs'])
                return infer(feed)
            policy.infer = capture_infer
            command = policy.act(state, refs)
            policy.infer = infer
            compare('action', policy.last, env.last_action.reshape(-1))
            compare('motor_target', np.asarray(command.joint_position_target_rad)[canonical][policy_to_world], capture['target'])
            compare('kp', np.asarray(command.kp_nm_rad)[canonical][policy_to_world], policy.kp)
            compare('kd', np.asarray(command.kd_nm_s_rad)[canonical][policy_to_world], policy.kd)
            for key, h in policy.history.items():
                compare('history', h, env.observations['policy'].funcs[key+'_history'].history)
            case_count += 1
        raw = np.linspace(-4, 4, 29, dtype=np.float32)[None]
        env.policy = lambda _: raw
        env._rl_step_scheduled()
        infer = policy.infer
        policy.infer = lambda _: {'action': raw}
        command = policy.act(state, refs)
        policy.infer = infer
        compare('forced_clip_slew', np.asarray(command.joint_position_target_rad)[canonical][policy_to_world], capture['target'])
    limits = {k: 2e-4 for k in errors}
    for key in ('reset_history', 'kp', 'kd'):
        limits[key] = 0.
    report = {'passed': all(errors[k] <= limits[k] for k in errors), 'cases': case_count, 'resets': 4,
        'endpoint_cases': endpoint_count, 'max_abs_error': errors, 'limits': limits,
        'deploy_commit': pins['deploy_commit'], 'training_commit': pins['training_commit'],
        'checkpoint_sha256': runtime['weights_sha256'], 'encoder_sha256': runtime['artifacts']['encoder']['sha256'],
        'scope': 'Author reference observation functions, released backward encoder, native ObsGroup and deployment step including action clipping, rescaling, target clipping and slew limits. Shuffled SDK joints, near-vertical rotations, zero history resets, truncated windows and forced out-of-range actions.',
        'benchmark_boundary': 'Three targets start at t+20ms. A fourth timing probe identifies the endpoint but is not encoded. Native endpoint-exclusive windows shrink; the final context is held to finish the canonical clip, without viewer wraparound.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
