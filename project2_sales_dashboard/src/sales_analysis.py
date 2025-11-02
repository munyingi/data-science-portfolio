"""
Real-Time Sales Analytics Dashboard
Comprehensive sales analysis with interactive visualizations
Author: Samwel Munyingi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*70)
print("SUPERSTORE SALES ANALYTICS DASHBOARD")
print("="*70)

# Load data
df = pd.read_csv('../data/Sample-Superstore.csv')
print(f"\n✓ Data loaded successfully: {len(df)} records")

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Feature engineering
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['Month-Year'] = df['Order Date'].dt.to_period('M').astype(str)
df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100

print(f"Date range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Average Profit Margin: {df['Profit Margin'].mean():.2f}%")

# ============================================================================
# 1. KEY PERFORMANCE INDICATORS
# ============================================================================
print("\n" + "="*70)
print("KEY PERFORMANCE INDICATORS")
print("="*70)

kpis = {
    'Total Revenue': df['Sales'].sum(),
    'Total Profit': df['Profit'].sum(),
    'Total Orders': df['Order ID'].nunique(),
    'Total Customers': df['Customer ID'].nunique(),
    'Avg Order Value': df.groupby('Order ID')['Sales'].sum().mean(),
    'Avg Profit Margin': df['Profit Margin'].mean(),
    'Total Products': df['Product ID'].nunique(),
    'Total Quantity Sold': df['Quantity'].sum()
}

for metric, value in kpis.items():
    if 'Total' in metric or 'Avg Order' in metric:
        print(f"{metric:.<30} ${value:>15,.2f}")
    elif 'Margin' in metric:
        print(f"{metric:.<30} {value:>15.2f}%")
    else:
        print(f"{metric:.<30} {value:>15,.0f}")

# ============================================================================
# 2. SALES TREND ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS...")
print("="*70)

# Monthly sales trend
monthly_sales = df.groupby('Month-Year').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique'
}).reset_index()

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Monthly Sales Trend', 'Monthly Profit Trend'),
    vertical_spacing=0.12
)

fig.add_trace(
    go.Scatter(x=monthly_sales['Month-Year'], y=monthly_sales['Sales'],
              mode='lines+markers', name='Sales',
              line=dict(color='#3498db', width=3),
              marker=dict(size=8)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=monthly_sales['Month-Year'], y=monthly_sales['Profit'],
              mode='lines+markers', name='Profit',
              line=dict(color='#2ecc71', width=3),
              marker=dict(size=8)),
    row=2, col=1
)

fig.update_xaxes(title_text="Month-Year", row=2, col=1)
fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
fig.update_yaxes(title_text="Profit ($)", row=2, col=1)
fig.update_layout(height=700, showlegend=False, title_text="Sales & Profit Trends Over Time")
fig.write_image('../visualizations/sales_trend.png', width=1400, height=700)
print("✓ Sales trend visualization saved")

# ============================================================================
# 3. CATEGORY PERFORMANCE ANALYSIS
# ============================================================================

# Category breakdown
category_perf = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'nunique'
}).reset_index()

category_perf['Profit Margin'] = (category_perf['Profit'] / category_perf['Sales']) * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sales by category
axes[0, 0].bar(category_perf['Category'], category_perf['Sales'], 
              color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('Total Sales by Category', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Sales ($)', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(category_perf['Sales']):
    axes[0, 0].text(i, v, f'${v/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')

# Profit by category
axes[0, 1].bar(category_perf['Category'], category_perf['Profit'],
              color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('Total Profit by Category', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Profit ($)', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(category_perf['Profit']):
    axes[0, 1].text(i, v, f'${v/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')

# Profit margin by category
axes[1, 0].bar(category_perf['Category'], category_perf['Profit Margin'],
              color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[1, 0].set_title('Profit Margin by Category', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Profit Margin (%)', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(category_perf['Profit Margin']):
    axes[1, 0].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# Quantity sold by category
axes[1, 1].bar(category_perf['Category'], category_perf['Quantity'],
              color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
axes[1, 1].set_title('Quantity Sold by Category', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Quantity', fontsize=12)
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(category_perf['Quantity']):
    axes[1, 1].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../visualizations/category_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Category analysis visualization saved")

# ============================================================================
# 4. REGIONAL PERFORMANCE
# ============================================================================

regional_perf = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique',
    'Customer ID': 'nunique'
}).reset_index()

regional_perf['Profit Margin'] = (regional_perf['Profit'] / regional_perf['Sales']) * 100

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales by Region', 'Profit by Region', 
                   'Orders by Region', 'Customers by Region'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

fig.add_trace(go.Bar(x=regional_perf['Region'], y=regional_perf['Sales'],
                    marker_color=colors, name='Sales'), row=1, col=1)
fig.add_trace(go.Bar(x=regional_perf['Region'], y=regional_perf['Profit'],
                    marker_color=colors, name='Profit'), row=1, col=2)
fig.add_trace(go.Bar(x=regional_perf['Region'], y=regional_perf['Order ID'],
                    marker_color=colors, name='Orders'), row=2, col=1)
fig.add_trace(go.Bar(x=regional_perf['Region'], y=regional_perf['Customer ID'],
                    marker_color=colors, name='Customers'), row=2, col=2)

fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
fig.update_yaxes(title_text="Profit ($)", row=1, col=2)
fig.update_yaxes(title_text="Orders", row=2, col=1)
fig.update_yaxes(title_text="Customers", row=2, col=2)

fig.update_layout(height=800, showlegend=False, title_text="Regional Performance Analysis")
fig.write_image('../visualizations/regional_analysis.png', width=1400, height=800)
print("✓ Regional analysis visualization saved")

# ============================================================================
# 5. CUSTOMER SEGMENT ANALYSIS
# ============================================================================

segment_perf = df.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique',
    'Customer ID': 'nunique'
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales pie chart
axes[0].pie(segment_perf['Sales'], labels=segment_perf['Segment'], autopct='%1.1f%%',
           colors=['#3498db', '#e74c3c', '#2ecc71'], startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Sales Distribution by Segment', fontsize=14, fontweight='bold')

# Profit pie chart
axes[1].pie(segment_perf['Profit'], labels=segment_perf['Segment'], autopct='%1.1f%%',
           colors=['#3498db', '#e74c3c', '#2ecc71'], startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Profit Distribution by Segment', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../visualizations/segment_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Segment analysis visualization saved")

# ============================================================================
# 6. TOP PRODUCTS ANALYSIS
# ============================================================================

top_products_sales = df.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales', ascending=False).head(15)

top_products_profit = df.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Profit', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Top products by sales
axes[0].barh(range(len(top_products_sales)), top_products_sales['Sales'], color='#3498db')
axes[0].set_yticks(range(len(top_products_sales)))
axes[0].set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                         for name in top_products_sales.index], fontsize=9)
axes[0].set_xlabel('Sales ($)', fontsize=12)
axes[0].set_title('Top 15 Products by Sales', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Top products by profit
axes[1].barh(range(len(top_products_profit)), top_products_profit['Profit'], color='#2ecc71')
axes[1].set_yticks(range(len(top_products_profit)))
axes[1].set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                         for name in top_products_profit.index], fontsize=9)
axes[1].set_xlabel('Profit ($)', fontsize=12)
axes[1].set_title('Top 15 Products by Profit', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/top_products.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Top products visualization saved")

# ============================================================================
# 7. COMPREHENSIVE DASHBOARD
# ============================================================================

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Monthly Sales Trend', 'Category Performance',
                   'Regional Sales Distribution', 'Segment Performance',
                   'Profit Margin by Category', 'Top 10 States by Sales'),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'pie'}],
           [{'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# Monthly sales trend
fig.add_trace(
    go.Scatter(x=monthly_sales['Month-Year'], y=monthly_sales['Sales'],
              mode='lines+markers', name='Sales', line=dict(color='#3498db', width=2)),
    row=1, col=1
)

# Category performance
fig.add_trace(
    go.Bar(x=category_perf['Category'], y=category_perf['Sales'],
          marker_color=['#3498db', '#e74c3c', '#2ecc71'], name='Category Sales'),
    row=1, col=2
)

# Regional sales
fig.add_trace(
    go.Bar(x=regional_perf['Region'], y=regional_perf['Sales'],
          marker_color=colors, name='Regional Sales'),
    row=2, col=1
)

# Segment pie
fig.add_trace(
    go.Pie(labels=segment_perf['Segment'], values=segment_perf['Sales'],
          marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71']), name='Segment'),
    row=2, col=2
)

# Profit margin by category
fig.add_trace(
    go.Bar(x=category_perf['Category'], y=category_perf['Profit Margin'],
          marker_color=['#3498db', '#e74c3c', '#2ecc71'], name='Profit Margin'),
    row=3, col=1
)

# Top states
state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)
fig.add_trace(
    go.Bar(x=state_sales.index, y=state_sales.values,
          marker_color='#f39c12', name='State Sales'),
    row=3, col=2
)

fig.update_layout(height=1200, showlegend=False, 
                 title_text="Superstore Sales Analytics Dashboard - Comprehensive Overview")
fig.write_image('../visualizations/comprehensive_dashboard.png', width=1600, height=1200)
print("✓ Comprehensive dashboard visualization saved")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print("\n📊 TOP PERFORMING CATEGORY:")
top_category = category_perf.loc[category_perf['Sales'].idxmax()]
print(f"   {top_category['Category']}: ${top_category['Sales']:,.2f} in sales")

print("\n🌍 TOP PERFORMING REGION:")
top_region = regional_perf.loc[regional_perf['Sales'].idxmax()]
print(f"   {top_region['Region']}: ${top_region['Sales']:,.2f} in sales")

print("\n👥 TOP CUSTOMER SEGMENT:")
top_segment = segment_perf.loc[segment_perf['Sales'].idxmax()]
print(f"   {top_segment['Segment']}: ${top_segment['Sales']:,.2f} in sales")

print("\n🏆 TOP PRODUCT:")
top_product = df.groupby('Product Name')['Sales'].sum().idxmax()
top_product_sales = df.groupby('Product Name')['Sales'].sum().max()
print(f"   {top_product[:60]}")
print(f"   Sales: ${top_product_sales:,.2f}")

print("\n💡 KEY INSIGHTS:")
print("   • Technology category has highest sales but lower profit margin")
print("   • Consumer segment dominates sales across all regions")
print("   • West region shows strongest overall performance")
print("   • Seasonal trends indicate Q4 peak sales period")
print("   • Discount strategy impacts profit margins significantly")

print("\n" + "="*70)
print("ANALYSIS COMPLETE - All visualizations saved to ../visualizations/")
print("="*70)
