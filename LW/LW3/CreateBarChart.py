from matplotlib import pyplot as plt


def create_bar_chart(task_title, task_data1, task_data2, y_label):
    age_groups = task_data1.index
    min_value = task_data1[y_label]['min'].values
    mean_value = task_data1[y_label]['mean'].values
    max_value = task_data1[y_label]['max'].values
    min_value2 = task_data2[y_label]['min'].values
    mean_value2 = task_data2[y_label]['mean'].values
    max_value2 = task_data2[y_label]['max'].values

    x = range(len(age_groups))
    width = 0.2

    plt.bar(x, min_value, width, alpha=0.5, label='Minimum ' + y_label)
    plt.bar([val + width for val in x], mean_value, width, alpha=0.5, label='Mean ' + y_label)
    plt.bar([val + width * 2 for val in x], max_value, width, alpha=0.5, label='Maximum ' + y_label)
    plt.bar([val + width * 3 for val in x], min_value2, width, alpha=0.5, label='New Minimum ' + y_label)
    plt.bar([val + width * 4 for val in x], mean_value2, width, alpha=0.5, label='New Mean ' + y_label)
    plt.bar([val + width * 5 for val in x], max_value2, width, alpha=0.5, label='New Maximum ' + y_label)

    plt.ylabel(y_label)
    plt.xticks([val + width for val in x], age_groups)
    plt.legend()
    plt.title(task_title)