import matplotlib.pyplot as plt
import numpy as np


def draw_result(x_label, y_label, x_labels, title,
                value1=None, value2=None, value1_label=None, value2_label=None,
                values=None, value_labels=None):
    # Set the number of bars
    x = np.arange(len(x_labels))

    # Set the width of the bars
    width = 0.25

    # Create the bar chart
    fig, ax = plt.subplots()

    if values is None:
        bars1 = ax.bar(x - width/2, value1, width, label=value1_label, color='.7')
        bars2 = ax.bar(x + width/2, value2, width, label=value2_label, color='blue')
    else:
        for bar_index in range(len(values)):
            ax.bar(x - (len(values)/2.0 - bar_index) * width / len(values), values[bar_index + 1], width / len(values), label=value_labels[bar_index])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Show the plot
    plt.savefig(title + ".png", bbox_inches="tight")
    #plt.show()
    plt.close()
