import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text 

def Line_Plots():
    # List of (filename, color, label, marker)
    csv_files = [
        ('Gradient Based Results.csv', '#00B4D8', 'Gradient Based', 'o'),   # Circle
        ('CGA Results.csv', '#D62828', 'CGA', 's'),                         # Square
        ('Baldwinian Results.csv', '#8338EC', 'Baldwinian', 'D'),           # Diamond
        ('Lamarckian Results.csv', '#FFBE0B', 'Lamarckian', '^'),           # Triangle Up
        ('Adaptive Baldwinian Results.csv', "#1FC606", 'Adaptive Baldwinian', 'v'), # Triangle Down
        ('Adaptive Lamarckian Results.csv', "#0B40FF", 'Adaptive Lamarckian', 'P')  # Plus-filled
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    max_overall_percentage = 0
    texts = []

    for file_name, color, label, marker in csv_files:
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)

            failure_counts = df['Failure Deviation'].value_counts()
            counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)
            counts_df.sort_index(inplace=True)

            counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100
            counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()

            counts_df.loc[counts_df.index.min() - 1] = 0
            counts_df.sort_index(inplace=True)
            ax.set_xticks(np.arange(0.0, counts_df.index.max() + 1.0, 1.0))
            ax.set_xticklabels([f'{x:.1f}' for x in ax.get_xticks()])

            # Use marker shape in plotting
            ax.plot(
                counts_df.index,
                counts_df['Cumulative Failure Percentage'],
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2,
                label=label
            )

            for i, txt in enumerate(counts_df['Cumulative Failure Percentage']):
                text = ax.annotate(
                    int(txt),
                    (counts_df.index[i], txt),
                    va='bottom',
                    ha='center',
                    fontsize=10,
                    color=color,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
                )
                texts.append(text)

            max_overall_percentage = max(max_overall_percentage, counts_df['Cumulative Failure Percentage'].max())
        else:
            print(f"Data File {file_name} Not Found")

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_title('Failure Rate vs. Conductance Control Deviation')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_xlabel('Conductance Control Deviation')
    ax.legend(loc='upper left')

    ax.set_facecolor('#ededed')
    ax.grid(color='white')
    ax.set_ylim(0, 1.1 * max_overall_percentage)

    plt.tight_layout()
    plt.savefig('Line Plot.png', dpi=300)
    plt.show()


def bar_plot():
    csv_files = [
        ('Gradient Based Results.csv', '#cecece', 'Gradient Based'),
        ('CGA Results.csv', '#a559aa', 'CGA'),
        ('Baldwinian Results.csv', '#59a89c', 'Baldwinian'),
        ('Lamarckian Results.csv', '#f0c571', 'Lamarckian'),
        ('Adaptive Baldwinian Results.csv', "#e02b35", 'Adaptive Baldwinian'),
        ('Adaptive Lamarckian Results.csv', "#082a54", 'Adaptive Lamarckian')
    ]

    # distinct marker shapes per model
    markers = ['o', 's', '^', 'D', 'P', 'X']

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.1
    all_deviations = set()

    # Step 1: Collect all unique deviations
    for file_name, _, _ in csv_files:
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)
            all_deviations.update(df['Failure Deviation'].unique())

    all_deviations = sorted(list(all_deviations))
    x = np.arange(len(all_deviations))  # shared x-axis for all bars

    # Step 2: Plot each dataset using shared x
    for i, (file_name, color, label) in enumerate(csv_files):
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)
            counts = df['Failure Deviation'].value_counts().sort_index()
            percentages = (counts / counts.sum()) * 100
            cumulative = percentages.cumsum().reindex(all_deviations, fill_value=0)

            # Force 100% failure after the last recorded deviation
            last_dev = max(counts.index)
            last_idx = all_deviations.index(last_dev)
            for fill_idx in range(last_idx + 1, len(all_deviations)):
                cumulative.iloc[fill_idx] = 100.0

            offset_x = x + (i - len(csv_files)/2) * width

            # Draw bars
            ax.bar(offset_x, cumulative, width=width, 
                   color=color, edgecolor="black", alpha=0.6)

            # Overlay markers at bar tops
            ax.plot(offset_x, cumulative, 
                    marker=markers[i % len(markers)], 
                    color=color, markersize=8, linestyle="",
                    label=label)

            # Tiny marker if value is zero
            for xi, val in zip(offset_x, cumulative):
                if val == 0:
                    ax.plot(xi, 0.5, marker=markers[i % len(markers)], 
                            color=color, markersize=5)

        else:
            print(f"Data File {file_name} Not Found")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{val:.1f}" for val in all_deviations])
    ax.set_title('Failure Rate vs. Conductance Control Deviation')
    ax.set_ylabel('Cumulative Failure Rate (%)')
    ax.set_xlabel('Conductance Control Deviation')
    ax.legend(loc='upper left')
    ax.set_facecolor('#ededed')
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('Bar Plot 2.png', dpi=300)
    plt.show()


def single_line_plot(output_name, file_name):

    fig, ax = plt.subplots(figsize=(10, 6))
    max_overall_percentage = 0  # To track max y-limit
    texts = []  # List to store text objects

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name)

        failure_counts = df['Failure Deviation'].value_counts()
        counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)
        counts_df.sort_index(inplace=True)

        counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100
        counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()

        # Add a zero-starting deviation one below the smallest
        counts_df.loc[counts_df.index.min() - 1] = 0
        counts_df.sort_index(inplace=True)
        ax.set_xticks(np.arange(0.0, counts_df.index.max() + 1.0, 1.0))
        ax.set_xticklabels([f'{x:.1f}' for x in ax.get_xticks()])


        # Plotting
        ax.plot(counts_df.index, counts_df['Cumulative Failure Percentage'], marker='o')

        # Track max percentage for setting y-limit
        max_overall_percentage = max(max_overall_percentage, counts_df['Cumulative Failure Percentage'].max())

    else:
        print(f"Data File {file_name} Not Found")

    ax.set_title('Failure Rate vs. Conductance Control Deviation')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_xlabel('Conductance Control Deviation')

    # Fill under the line
    csv_file = file_name
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        failure_counts = df['Failure Deviation'].value_counts()
        counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)
        counts_df.sort_index(inplace=True)
        counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100
        counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()
        counts_df.loc[counts_df.index.min() - 1] = 0
        counts_df.sort_index(inplace=True)
        ax.fill_between(counts_df.index, counts_df['Cumulative Failure Percentage'], alpha=0.4)

    ax.set_facecolor('#ededed')
    ax.grid(color='white')

    ax.set_ylim(0, 1.1 * max_overall_percentage)  # Add 10% buffer

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    bar_plot()
