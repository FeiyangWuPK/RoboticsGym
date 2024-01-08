from scripts.train_newalgo import train_cassie_v4, train_cassie_v5, test_student_policy
from roboticsgym.utils.gen_experts import obtain_ik_data_in_replay_buffer

if __name__ == "__main__":
    train_cassie_v5()
    # obtain_ik_data_in_replay_buffer(n_traj=10)
    # test_student_policy()
