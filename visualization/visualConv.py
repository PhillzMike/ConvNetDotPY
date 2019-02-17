import matplotlib.pyplot as plt


def visualize_layer(layer, index=0):
    assert 0 <= index <= layer.shape[0]
    fig = plt.figure(figsize=(20, 20))
    n_filters = layer._inp.shape[3]
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow((layer._inp[index, :, :, i]), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))
