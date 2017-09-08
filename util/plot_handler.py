from matplotlib import pyplot as plt


'''Create and show plot'''


def show_plot(data_1, data_2, target, title):
    plt.plot(data_1)
    plt.plot(data_2)
    plt.title(title)
    plt.ylabel(target)
    plt.xlabel('Timestep')
    plt.legend(['True', 'Predicted'], loc = 'upper right')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


'''Save plot to PNG file'''


def save_plot(output_dir, job_name, file_name, target, name, *data, run = -1):
    for series in data:
        plt.plot(series)
    plt.title('Model predictions')
    plt.ylabel(target)
    plt.xlabel('Timestep')
    plt.legend(['True', 'Predicted'], loc = 'upper right')
    directory = '%s/%d/%s/%s' % (output_dir, job_name, file_name, target)
    if run > -1:
        directory = '%s/%d' % (directory, run)
    directory = '%s/%s.png' % (directory, name)
    plt.savefig(directory, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
