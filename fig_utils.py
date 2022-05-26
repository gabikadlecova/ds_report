# +
import matplotlib.pyplot as plt
import seaborn as sns


def rotate_labels(ax):
    plt.draw()
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


def plot_baseline(ax, baselines, task_name):
    palette = sns.color_palette()

    for b, col in zip(baselines, palette):
        val = float(b[task_name])
        ax.hlines(val, 'PCA_2-RF_50', 'PCA_20-RF_500', color=col, linestyle='--')
    

def draw_metrics(df, task_1, task_2, baselines, plot_1=True, plot_2=True, title=None):
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    if title is not None:
        plt.suptitle(title)

    sns.lineplot(data=df, x='name', y=task_1, hue='task_name', ax=axs[0])
    rotate_labels(axs[0])
    
    if plot_1:
        plot_baseline(axs[0], baselines, task_1)

    sns.lineplot(data=df, x='name', y=task_2, hue='task_name', ax=axs[1])
    rotate_labels(axs[1])
    
    if plot_2:
        plot_baseline(axs[1], baselines, task_2)

    fig.tight_layout()
    plt.show()
    
    
def plot_final_results(data, task_1, task_2, second_hue=True, title=None, sharey=False):
    fix, axs = plt.subplots(1, 2, figsize=(7,5), sharey=sharey)
    if title is not None:
        plt.suptitle(title)

    second_hue = 'type' if second_hue else None
    sns.barplot(data=data, x='name', y=task_1, hue='type', ax=axs[0])
    axs[0].legend(loc='lower left')
    
    sns.barplot(data=data, x='name', y=task_2, hue=second_hue, ax=axs[1])
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    plt.show()
# -


