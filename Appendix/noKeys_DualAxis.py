import pandas as pd
import json
import matplotlib.pyplot as plt


with open(r'C:\Users\kstat\Documents\Dissertation\Data\datasetWithKeys.json', 'r', encoding='utf-8') as infile:
    dataset = json.load(infile)


with open(r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Co-Occurrence\co_occurrence_frequencies_by_year.json', 'r', encoding='utf-8') as infile:
    co_occurrence_data = json.load(infile)

# Extract the "no_keywords_count_by_year" from the loaded co-occurrence data
no_keywords_count_by_year = co_occurrence_data.get("no_keywords_count_by_year", {})


df = pd.DataFrame(dataset)


total_abstracts_by_year = df.groupby('Year').size().reset_index(name='Total_Abstracts')

# Create a DataFrame for the "no_keywords_count_by_year" from the co-occurrence data
no_keywords_df = pd.DataFrame(list(no_keywords_count_by_year.items()), columns=['Year', 'No_Keywords_Count'])

# Merge both DataFrames on the 'Year' column to combine the total abstracts and no keywords counts
merged_df = pd.merge(total_abstracts_by_year, no_keywords_df, on='Year', how='left')

# Calculate the proportion of abstracts without keywords for each year
merged_df['Proportion Without Terms'] = merged_df['No_Keywords_Count'] / merged_df['Total_Abstracts']

# Plotting the dual-axis chart
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the total abstracts on the first axis
ax1.bar(merged_df['Year'], merged_df['Total_Abstracts'], color='lightblue', label='Total Abstracts')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Abstracts', color='lightblue')
ax1.tick_params(axis='y', labelcolor='lightblue')

# Create the second axis to plot the proportion of abstracts without keywords
ax2 = ax1.twinx()
ax2.plot(merged_df['Year'], merged_df['Proportion Without Terms'], color='darkblue', marker='o', label='Proportion Without Terms')
ax2.set_ylabel('Proportion Without Terms', color='darkblue')
ax2.tick_params(axis='y', labelcolor='darkblue')

# Format the x-axis to display years as two digits 
ax1.set_xticks(merged_df['Year'])
ax1.set_xticklabels([f"'{str(year)[-2:]}" for year in merged_df['Year']], rotation=45)

# Add a title and a grid
plt.title('Total Abstracts and Proportion Without Terms per Year')
fig.tight_layout()


plt.savefig('C:/Users/kstat/Documents/Dissertation/Data/Methods/Method_2/Co-Occurrence/dual_axis_no_keywords_plot.png')


plt.show()
