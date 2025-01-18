import json
import plotly.graph_objects as go
import os


input_file_path = r"C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Co-Occurrence\co-occurrenceFreqsNormalizedTop30ByYear.json"
output_dir = r"C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Co-Occurrence\Plots"
os.makedirs(output_dir, exist_ok=True)


with open(input_file_path, 'r', encoding='utf-8') as f:
    results = json.load(f)


fig = go.Figure()

# Extract co-occurrence frequencies per year
for year in range(1995, 2025):
    year = str(year)
    year_data = results['co_occurrence_frequencies'].get(year, [])
    no_keywords_count = results['no_keywords_count_by_year'].get(year, 0)

    if year_data:  # Skip years with no data
        # Extract keyword pairs and their normalized co-occurrence frequencies
        keyword_pairs = [pair[0] for pair in year_data]
        co_occurrence_scores = [pair[1] for pair in year_data]

        # Create hover text
        hover_texts = [
            f"Year: {year}<br>Term Pair: {pair}<br>Normalized Co-occurrence Frequency: {score:.4f}<br>No Terms Count: {no_keywords_count}"
            for pair, score in zip(keyword_pairs, co_occurrence_scores)
        ]

        # Add bar trace for the year
        fig.add_trace(go.Bar(
            x=co_occurrence_scores,
            y=keyword_pairs,
            name=f"{year} (No Terms: {no_keywords_count})",
            orientation='h',
            hovertext=hover_texts,
            hoverinfo="text"
        ))

# Customize layout with exclusive toggle
fig.update_layout(
    title="Top 30 Term Pair Co-occurrence Frequencies by Year (1995–2024)",
    xaxis_title="Normalized Co-occurrence Frequency",
    yaxis_title="Term Pairs",
    barmode="overlay",  # Keeps the bars independent
    legend=dict(
        title="Years",
        traceorder="normal",
        itemclick="toggleothers",  # Enables exclusive display on click
        itemdoubleclick="toggle"  # Enables isolation on double-click
    ),
    template="plotly_white",
    height=1000,  # Adjust height for better visibility of all years
)


output_html_path = os.path.join(output_dir, "Interactive_BarPlot_Co-occurrenceFreqs_Top30_byYear.html")
fig.write_html(output_html_path)


fig.show()

print(f"Interactive plot saved at {output_html_path}")
