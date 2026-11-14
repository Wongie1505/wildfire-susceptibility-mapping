# import geopandas as gpd
# import rasterio
# import numpy as np
# import pandas as pd

# # ── PATHS ──────────────────────────────────────────────────────────────────
# DATA = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA"


# gpkg_path   = DATA + r"\Fire\Fire_non-fire_merge.gpkg"
# raster_path = DATA + r"\susceptibility_prob_Random_Forest.tif"

# # ── LOAD ALL POINTS ────────────────────────────────────────────────────────
# gdf = gpd.read_file(gpkg_path)
# print(f"Total points loaded: {len(gdf)}")
# print(f"Fire=1: {(gdf['Fire']==1).sum()}  |  Fire=0: {(gdf['Fire']==0).sum()}")

# # ── REPROJECT TO MATCH RASTER CRS ─────────────────────────────────────────
# with rasterio.open(raster_path) as src:
#     raster_crs = src.crs
#     nodata_val = src.nodata

# gdf = gdf.to_crs(raster_crs)

# # ── SAMPLE PROBABILITY RASTER AT ALL POINT LOCATIONS ──────────────────────
# coords = [(geom.x, geom.y) for geom in gdf.geometry]

# with rasterio.open(raster_path) as src:
#     sampled = [s[0] for s in src.sample(coords)]

# gdf['susceptibility_prob'] = sampled

# # Drop nodata (points that fell outside raster extent)
# if nodata_val is not None:
#     before = len(gdf)
#     gdf = gdf[gdf['susceptibility_prob'] != nodata_val]
#     print(f"Dropped {before - len(gdf)} nodata points")

# gdf = gdf[gdf['susceptibility_prob'].notna()]
# print(f"Points after nodata drop: {len(gdf)}")

# # ── BIN INTO SUSCEPTIBILITY CLASSES ───────────────────────────────────────
# bins   = [0.0, 0.33, 0.66, 1.0]
# labels = ['Low (0.00–0.33)', 'Medium (0.34–0.66)', 'High (0.67–1.00)']

# gdf['susc_class'] = pd.cut(
#     gdf['susceptibility_prob'],
#     bins=bins,
#     labels=labels,
#     include_lowest=True
# )

# # ── CROSS-TABULATION ───────────────────────────────────────────────────────
# fire1 = gdf[gdf['Fire'] == 1]
# fire0 = gdf[gdf['Fire'] == 0]

# counts_1 = fire1['susc_class'].value_counts().reindex(labels, fill_value=0)
# counts_0 = fire0['susc_class'].value_counts().reindex(labels, fill_value=0)

# pct_1 = (counts_1 / counts_1.sum() * 100).round(2)
# pct_0 = (counts_0 / counts_0.sum() * 100).round(2)

# # ── BUILD RESULTS TABLE ────────────────────────────────────────────────────
# results = pd.DataFrame({
#     'Susceptibility Class': labels,
#     'Fire=1 Count':         counts_1.values,
#     'Fire=1 %':             pct_1.values,
#     'Fire=0 Count':         counts_0.values,
#     'Fire=0 %':             pct_0.values,
# })

# print("\n──────────────────────────────────────────────────────────────")
# print("         SPATIAL VALIDATION SUMMARY")
# print("──────────────────────────────────────────────────────────────")
# print(results.to_string(index=False))

# # ── KEY THESIS METRICS ─────────────────────────────────────────────────────
# high_med_fire1 = pct_1[labels[1]] + pct_1[labels[2]]   # Medium + High for fire=1
# low_fire0      = pct_0[labels[0]]                        # Low for fire=0

# print("\n── Key Validation Metrics ────────────────────────────────────")
# print(f"Fire=1 points in Medium+High zones : {high_med_fire1:.2f}%")
# print(f"Fire=0 points in Low zone          : {low_fire0:.2f}%")

# if high_med_fire1 >= 70:
#     print(f"✓ Model correctly predicts {high_med_fire1:.2f}% of actual fire locations as Medium–High susceptibility")
# else:
#     print(f"✗ {high_med_fire1:.2f}% — below 70% threshold, worth discussing in limitations")


# # ── SAVE CSV ───────────────────────────────────────────────────────────────
# out_path =r"spatial_validation_results.csv"

# # Build the formatted table as a string then write it
# lines = []
# lines.append("SPATIAL VALIDATION SUMMARY")
# lines.append("─" * 62)
# lines.append(f"{'Susceptibility Class':>22}  {'Fire=1 Count':>12}  {'Fire=1 %':>8}  {'Fire=0 Count':>12}  {'Fire=0 %':>8}")

# for label in labels:
#     lines.append(
#         f"{label:>22}  {counts_1[label]:>12}  {pct_1[label]:>8.2f}  {counts_0[label]:>12}  {pct_0[label]:>8.2f}"
#     )

# lines.append("")
# lines.append("── Key Validation Metrics " + "─" * 36)
# lines.append(f"Fire=1 points in Medium+High zones : {high_med_fire1:.2f}%")
# lines.append(f"Fire=0 points in Low zone          : {low_fire0:.2f}%")

# if high_med_fire1 >= 70:
#     lines.append(f"✓ Model correctly predicts {high_med_fire1:.2f}% of actual fire locations as Medium–High susceptibility")
# else:
#     lines.append(f"✗ {high_med_fire1:.2f}% — below 70% threshold, worth discussing in limitations")

# with open(out_path, 'w', encoding='utf-8') as f:
#     f.write('\n'.join(lines))

# print(f"\nSaved → {out_path}")




import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PATHS ──────────────────────────────────────────────────────────────────
DATA = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA"

gpkg_path   = DATA + r"\Fire\Fire_non-fire_merge.gpkg"
raster_path = DATA + r"\susceptibility_prob_Random_Forest.tif"

# ── LOAD ALL POINTS ────────────────────────────────────────────────────────
gdf = gpd.read_file(gpkg_path)
print(f"Total points loaded: {len(gdf)}")
print(f"Fire=1: {(gdf['Fire']==1).sum()}  |  Fire=0: {(gdf['Fire']==0).sum()}")

# ── REPROJECT TO MATCH RASTER CRS ─────────────────────────────────────────
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    nodata_val = src.nodata

gdf = gdf.to_crs(raster_crs)

# ── SAMPLE PROBABILITY RASTER ──────────────────────────────────────────────
coords = [(geom.x, geom.y) for geom in gdf.geometry]

with rasterio.open(raster_path) as src:
    sampled = [s[0] for s in src.sample(coords)]

gdf['susceptibility_prob'] = sampled

if nodata_val is not None:
    before = len(gdf)
    gdf = gdf[gdf['susceptibility_prob'] != nodata_val]
    print(f"Dropped {before - len(gdf)} nodata points")

gdf = gdf[gdf['susceptibility_prob'].notna()]
print(f"Points after nodata drop: {len(gdf)}")

# ── BIN INTO SUSCEPTIBILITY CLASSES ───────────────────────────────────────
bins   = [0.0, 0.33, 0.66, 1.0]
labels = ['Low (0.00–0.33)', 'Medium (0.34–0.66)', 'High (0.67–1.00)']
colors = ['#2ecc71', '#f39c12', '#e74c3c']   # green / orange / red

gdf['susc_class'] = pd.cut(
    gdf['susceptibility_prob'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# ── CROSS-TABULATION ───────────────────────────────────────────────────────
fire1 = gdf[gdf['Fire'] == 1]
fire0 = gdf[gdf['Fire'] == 0]

counts_1 = fire1['susc_class'].value_counts().reindex(labels, fill_value=0)
counts_0 = fire0['susc_class'].value_counts().reindex(labels, fill_value=0)

pct_1 = (counts_1 / counts_1.sum() * 100).round(2)
pct_0 = (counts_0 / counts_0.sum() * 100).round(2)

# ── CONSOLE PRINT ─────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Susceptibility Class': labels,
    'Fire=1 Count':         counts_1.values,
    'Fire=1 %':             pct_1.values,
    'Fire=0 Count':         counts_0.values,
    'Fire=0 %':             pct_0.values,
})

print("\n──────────────────────────────────────────────────────────────")
print("         SPATIAL VALIDATION SUMMARY")
print("──────────────────────────────────────────────────────────────")
print(results.to_string(index=False))

high_med_fire1 = pct_1[labels[1]] + pct_1[labels[2]]
low_fire0      = pct_0[labels[0]]

print("\n── Key Validation Metrics ────────────────────────────────────")
print(f"Fire=1 points in Medium+High zones : {high_med_fire1:.2f}%")
print(f"Fire=0 points in Low zone          : {low_fire0:.2f}%")

if high_med_fire1 >= 70:
    print(f"✓ Model correctly predicts {high_med_fire1:.2f}% of actual fire locations as Medium–High susceptibility")
else:
    print(f"✗ {high_med_fire1:.2f}% — below 70% threshold, worth discussing in limitations")

# ── SAVE STYLED EXCEL ─────────────────────────────────────────────────────
out_xlsx = r"spatial_validation_results.xlsx"

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # Write a blank sheet we'll populate manually
    results.to_excel(writer, sheet_name='Validation', index=False, startrow=3, startcol=0)

    wb = writer.book
    ws = writer.sheets['Validation']

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # ── Title row ─────────────────────────────────────────────────────────
    ws.merge_cells('A1:E1')
    title_cell = ws['A1']
    title_cell.value = 'SPATIAL VALIDATION SUMMARY'
    title_cell.font      = Font(bold=True, size=13)
    title_cell.alignment = Alignment(horizontal='center')

    # ── Separator row 2 ───────────────────────────────────────────────────
    ws.merge_cells('A2:E2')
    ws['A2'].value = '─' * 62

    # ── Header row (row 4 = startrow 3 + 1) ──────────────────────────────
    header_fill = PatternFill(fill_type='solid', fgColor='2C3E50')
    header_font = Font(bold=True, color='FFFFFF')

    for col in range(1, 6):
        cell = ws.cell(row=4, column=col)
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = Alignment(horizontal='center')

    # ── Row fills by susceptibility class ────────────────────────────────
    row_colors = {
        'Low (0.00–0.33)':    'D5F5E3',   # light green
        'Medium (0.34–0.66)': 'FDEBD0',   # light orange
        'High (0.67–1.00)':   'FADBD8',   # light red
    }

    for i, label in enumerate(labels):
        excel_row = 5 + i   # data starts at row 5
        fill = PatternFill(fill_type='solid', fgColor=row_colors[label])
        for col in range(1, 6):
            cell = ws.cell(row=excel_row, column=col)
            cell.fill      = fill
            cell.alignment = Alignment(horizontal='center')

    # ── Key metrics block ─────────────────────────────────────────────────
    metrics_start = 9

    ws.merge_cells(f'A{metrics_start}:E{metrics_start}')
    ws[f'A{metrics_start}'].value = '── Key Validation Metrics ' + '─' * 30
    ws[f'A{metrics_start}'].font  = Font(bold=True)

    ws.merge_cells(f'A{metrics_start+1}:E{metrics_start+1}')
    ws[f'A{metrics_start+1}'].value = f'Fire=1 points in Medium+High zones : {high_med_fire1:.2f}%'

    ws.merge_cells(f'A{metrics_start+2}:E{metrics_start+2}')
    ws[f'A{metrics_start+2}'].value = f'Fire=0 points in Low zone          : {low_fire0:.2f}%'

    ws.merge_cells(f'A{metrics_start+3}:E{metrics_start+3}')
    verdict = (
        f'✓ Model correctly predicts {high_med_fire1:.2f}% of actual fire locations as Medium–High susceptibility'
        if high_med_fire1 >= 70
        else f'✗ {high_med_fire1:.2f}% — below 70% threshold, worth discussing in limitations'
    )
    verdict_cell = ws[f'A{metrics_start+3}']
    verdict_cell.value = verdict
    verdict_cell.font  = Font(bold=True, color='1E8449' if high_med_fire1 >= 70 else 'C0392B')

    # ── Column widths ─────────────────────────────────────────────────────
    col_widths = [22, 14, 10, 14, 10]
    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

print(f"\nSaved Excel → {out_xlsx}")

# ── PIE CHART: FIRE = 1 ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 7))

wedges1, texts1, autotexts1 = ax1.pie(
    counts_1.values,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=11)
)

for at in autotexts1:
    at.set_fontsize(10)
    at.set_fontweight('bold')
    at.set_color('white')

ax1.set_title(
    'Susceptibility Class Distribution\nFire Occurrence Points (Fire = 1)',
    fontsize=13, fontweight='bold', pad=20
)

# Add count labels to legend
legend_labels1 = [f'{l}  (n={c:,})' for l, c in zip(labels, counts_1.values)]
patches1 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels1)]
ax1.legend(handles=patches1, loc='lower center', bbox_to_anchor=(0.5, -0.12),
           fontsize=10, frameon=False)

plt.tight_layout()
pie1_path = r"pie_fire1_susceptibility.png"
plt.savefig(pie1_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {pie1_path}")

# ── PIE CHART: FIRE = 0 ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 7))

wedges2, texts2, autotexts2 = ax2.pie(
    counts_0.values,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=11)
)

for at in autotexts2:
    at.set_fontsize(10)
    at.set_fontweight('bold')
    at.set_color('white')

ax2.set_title(
    'Susceptibility Class Distribution\nNon-Fire Points (Fire = 0)',
    fontsize=13, fontweight='bold', pad=20
)

legend_labels0 = [f'{l}  (n={c:,})' for l, c in zip(labels, counts_0.values)]
patches0 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels0)]
ax2.legend(handles=patches0, loc='lower center', bbox_to_anchor=(0.5, -0.12),
           fontsize=10, frameon=False)

plt.tight_layout()
pie0_path = r"pie_fire0_susceptibility.png"
plt.savefig(pie0_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {pie0_path}")

print("\n── All outputs saved ─────────────────────────────────────────")
print(f"  Excel  : {out_xlsx}")
print(f"  Pie 1  : {pie1_path}")
print(f"  Pie 0  : {pie0_path}")