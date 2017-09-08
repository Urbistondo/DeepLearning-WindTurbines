from matplotlib import pyplot as plt


def show_plot(y, predictions, target, title):
    plt.plot(y)
    plt.plot(predictions)
    plt.title(title)
    plt.ylabel(target)
    plt.xlabel('Timestep')
    plt.legend(['True', 'Predicted'], loc = 'upper right')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


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
