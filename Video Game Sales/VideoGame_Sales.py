import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
data = pd.read_csv("vgsales.csv")

# Increase figure size for better readability
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")  # Professional theme

# Function to plot pie chart of global sales by genre
def plot_genre_pie():
    genre_sales = data.groupby("Genre")["Global_Sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("pastel", len(genre_sales))
    ax.pie(genre_sales, labels=genre_sales.index, autopct="%1.1f%%", colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Global Video Game Sales by Genre")
    show_plot(fig)

# Function to plot pie chart of global sales by platform (Top 11 + Others)
def plot_platform_pie():
    platform_sales = data.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
    top_11 = platform_sales[:11]
    others = platform_sales[11:].sum()
    top_11["Others"] = others
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(top_11))
    ax.pie(top_11, labels=top_11.index, autopct="%1.1f%%", colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Global Video Game Sales by Platform (Top 11 + Others)")
    show_plot(fig)

# Function to plot bar graph of top 10 selling video games
def plot_top_games():
    top_games = data.nlargest(10, "Global_Sales")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=top_games["Name"], x=top_games["Global_Sales"], palette="viridis", ax=ax)
    ax.set_xlabel("Global Sales (in Millions)")
    ax.set_ylabel("Game Name")
    ax.set_title("Top 10 Best-Selling Video Games")
    show_plot(fig)

# Function to plot bar graph of top 10 publishers by global sales
def plot_top_publishers():
    publisher_sales = data.groupby("Publisher")["Global_Sales"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=publisher_sales.index, x=publisher_sales.values, palette="mako", ax=ax)
    ax.set_xlabel("Global Sales (in Millions)")
    ax.set_ylabel("Publisher")
    ax.set_title("Top 10 Video Game Publishers by Global Sales")
    show_plot(fig)

# Function to plot bar graph of video game sales by genre across different regions
def plot_genre_by_region():
    genre_sales = data.groupby("Genre")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    genre_sales.plot(kind="bar", stacked=True, colormap="Set2", ax=ax, edgecolor="black")
    ax.set_ylabel("Sales (in Millions)")
    ax.set_title("Video Game Sales by Genre Across Different Regions")
    plt.xticks(rotation=45)
    show_plot(fig)

# Function to plot line graph of global video game sales trend over the years
def plot_sales_trend():
    yearly_sales = data.groupby("Year")["Global_Sales"].sum().dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker="o", color="b", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Global Sales (in Millions)")
    ax.set_title("Global Video Game Sales Trend Over the Years")
    show_plot(fig)

# Function to display the selected plot
def show_plot(fig):
    global canvas
    for widget in frame.winfo_children():  # Clear previous plots
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# GUI using tkinter
root = tk.Tk()
root.title("Video Game Sales Visualization")
root.geometry("1200x700")  # Set bigger screen size

# Dropdown Menu for selecting graphs
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

def update_plot(choice):
    if choice == "Global Sales by Genre (Pie Chart)":
        plot_genre_pie()
    elif choice == "Global Sales by Platform (Pie Chart)":
        plot_platform_pie()
    elif choice == "Top 10 Selling Games (Bar Graph)":
        plot_top_games()
    elif choice == "Top 10 Publishers (Bar Graph)":
        plot_top_publishers()
    elif choice == "Genre Sales by Region (Bar Graph)":
        plot_genre_by_region()
    elif choice == "Global Sales Trend (Line Graph)":
        plot_sales_trend()

options = [
    "Global Sales by Genre (Pie Chart)",
    "Global Sales by Platform (Pie Chart)",
    "Top 10 Selling Games (Bar Graph)",
    "Top 10 Publishers (Bar Graph)",
    "Genre Sales by Region (Bar Graph)",
    "Global Sales Trend (Line Graph)"
]

dropdown = ttk.Combobox(root, values=options, state="readonly", width=40)
dropdown.set("Select a Visualization")
dropdown.pack(pady=20)

dropdown.bind("<<ComboboxSelected>>", lambda event: update_plot(dropdown.get()))

root.mainloop()
