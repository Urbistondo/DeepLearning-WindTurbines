from matplotlib import pyplot as plt


def show_plot(y, predictions):
    plt.plot(y)
    plt.plot(predictions)
    plt.title('Model predictions')
    plt.ylabel('WGENBearNDETemp')
    plt.xlabel('Timestep')
    plt.legend(['True', 'Predicted'], loc = 'upper left')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


def save_plot(output_dir, job_name, file_name, name, *data):
    for series in data:
        plt.plot(series)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc = 'upper right')
    plt.figure(figsize = (8.0, 5.0))
    plt.savefig('%s/%d/%s/%s.png' % (output_dir, job_name, file_name, name),
                bbox_inches = 'tight', pad_inches = 0)
    plt.close()
