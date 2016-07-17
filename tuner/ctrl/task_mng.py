task_cnt = 0


def init_task():
    global task_cnt
    task_cnt = 0


def insert_task(hyper_dict, cur_loss):
    global task_cnt
    task_cnt += 1


def view_task():
    global task_cnt
    print(task_cnt)

if __name__ == '__main__':
    init_task()
    insert_task(0, 0)
    view_task()

