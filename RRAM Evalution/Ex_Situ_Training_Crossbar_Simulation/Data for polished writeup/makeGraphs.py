import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text  # Import adjustText

def evaluation_Plots():
    # List of (filename, color, label)
    csv_files = [
        ('Gradient Based Results.csv', '#00B4D8', 'Gradient Based'),
        ('CGA Results.csv', '#D62828', 'CGA'),
        ('Baldwinian Results.csv', '#8338EC', 'Baldwinian'),
        ('Lamarckian Results.csv', '#FFBE0B', 'Lamarckian')
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    max_overall_percentage = 0  # To track max y-limit
    texts = []  # List to store text objects

    for file_name, color, label in csv_files:
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
            ax.plot(counts_df.index, counts_df['Cumulative Failure Percentage'], color=color, marker='o', label=label)

            # Improved annotations
            for i, txt in enumerate(counts_df['Cumulative Failure Percentage']):
                # Add the text annotation to the list for adjustment
                text = ax.annotate(
                    int(txt),
                    (counts_df.index[i], txt),
                    va='bottom',  # Position label above the point
                    ha='center',
                    fontsize=10,  # Increase font size
                    color=color,  # Match label color with the line
                    fontweight='bold',  # Make the font bold
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')  # Add a background to improve visibility
                )
                texts.append(text)  # Append the text object to the list

            # Track max percentage for setting y-limit
            max_overall_percentage = max(max_overall_percentage, counts_df['Cumulative Failure Percentage'].max())

        else:
            print(f"Data File {file_name} Not Found")

    # Adjust the label positions to avoid overlap using adjustText
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))  # Use arrows for better visibility

    ax.set_title('Failure Rate vs. Conductance Control Deviation')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_xlabel('Conductance Control Deviation')
    ax.legend(loc='upper left')

    # Fill under the line
    for file_name, color, label in csv_files:
        csv_file = f'{file_name}'
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            failure_counts = df['Failure Deviation'].value_counts()
            counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)
            counts_df.sort_index(inplace=True)
            counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100
            counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()
            counts_df.loc[counts_df.index.min() - 1] = 0
            counts_df.sort_index(inplace=True)
            ax.fill_between(counts_df.index, counts_df['Cumulative Failure Percentage'], color=color, alpha=0.4)

    ax.set_facecolor('#ededed')
    ax.grid(color='white')

    ax.set_ylim(0, 1.1 * max_overall_percentage)  # Add 10% buffer

    plt.tight_layout()
    plt.savefig('Graph with shade fill.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    evaluation_Plots()
