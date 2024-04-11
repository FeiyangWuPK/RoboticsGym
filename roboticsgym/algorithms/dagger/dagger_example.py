from DAGGER import Dagger

dagger = Dagger(
    num_envs=number_of_envs,  # this environments are step-wise parallelized. You can think of them like running mujoco environments in parallel
    num_transitions_per_env=n_steps,  # this is the number of steps that each environment will take for each trajectory
    obs_shape=observation_shape,  # this is the shape of the observation
    action_shape=action_shape,  # this is the shape of the action(network output shape)
    device=device,  # torch device
    architecture=module,  # network architecture
    learning_rate=learning_Rate,
    num_learning_epochs=learning_epoch_num,  # learning epochs
    num_mini_batches=1,  # this should start from 1. The storage will gradually grow as it data accumulated
    log_dir=saver.data_dir,
)  # for logging

for update in range(100 + 1):
    # training time
    training_start = time.time()
    env.reset()
    for step in range(n_steps):
        observation = env.observe()
        network_action = module.architecture(torch.from_numpy(observation).to("cuda"))
        action, dones = env.net_step(
            network_action.detach().to("cpu").numpy()
        )  # the action is action label for the DAgger.
        dagger.add_transitions(
            observation, action
        )  # the shape= (num_envs, *obs_shape) and (num_envs, *action_shape)

    dagger.update(update=update)  # training happens here

    training_end = time.time()
    torch.save(
        module.state_dict(), dagger.net_save_dir + "/MPC_actor_{}.pt".format(update)
    )  # model saving
