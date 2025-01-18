import json
import plotly.graph_objects as go
import os


output_path = r"C:\\Users\\kstat\\Documents\\Dissertation\\Data\\Methods\\Method_2\\Keywords(Direct)\\THIS\\All_Years_Top_Keywords.json"
output_dir = r"C:\\Users\\kstat\\Documents\\Dissertation\\Data\\Methods\\Method_2\\Keywords(Direct)\\THIS\\Plots"
os.makedirs(output_dir, exist_ok=True)


with open(output_path, 'r') as f:
    results = json.load(f)


fig = go.Figure()

# Extract years and data
for year in range(1995, 2025):
    year = str(year)
    terms_data = results.get(year, {})
    total_abstracts = results.get(f"{year}_total_abstracts", 0)

    if terms_data:  # Skip years with no data
        terms = list(terms_data.keys())
        scores = list(terms_data.values())

        # Create hover text
        hover_texts = [
            f"Year: {year}<br>Term: {term}<br>TF-IDF Score: {score:.4f}<br>Total Abstracts: {total_abstracts}"
            for term, score in zip(terms, scores)
        ]

        # Add bar trace for the year
        fig.add_trace(go.Bar(
            x=scores,
            y=terms,
            name=f"{year} (Total Abstracts: {total_abstracts})",
            orientation='h',
            hovertext=hover_texts,
            hoverinfo="text"
        ))

# Customize layout with exclusive toggle
fig.update_layout(
    title="Top 30 Terms by Year (1995–2024)",
    xaxis_title="TF-IDF Score",
    yaxis_title="Terms",
    barmode="overlay",  # Keeps the bars independent
    legend=dict(
        title="Years",
        traceorder="normal",
        itemclick="toggleothers",  # Enables exclusive display on click
        itemdoubleclick="toggle"  # Enables isolation on double-click
    ),
    template="plotly_white",
    height=1000,  
)


output_html_path = os.path.join(output_dir, "Interactive_Top_Keywords_By_Year.html")
fig.write_html(output_html_path)


fig.show()

print(f"Interactive plot saved at {output_html_path}")
