import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from flask import Flask, request, send_file, render_template_string
import io
import json


# =========================
# Helper Functions
# =========================

def generate_image_url(pokemon, mega=False, paldean=False, galar=False, crown=False):
    strr = pokemon.lower()
    if mega:
        strr += "-mega"
    elif paldean:
        strr += "-paldea"
    elif galar:
        strr += "-galar"
    elif crown:
        strr += "-crowned"
    strr = "https://r2.limitlesstcg.net/pokemon/gen9/" + strr + ".png"
    return strr

def parse_color(value):
    """Convert Excel color field to a matplotlib-compatible color."""
    if pd.isna(value):
        return None  # default color
    value = str(value).strip()

    # Convert rgb(...) format to matplotlib-usable tuple
    if value.lower().startswith("rgb"):
        try:
            nums = value[value.find("(")+1:value.find(")")].split(",")
            r, g, b = [int(x)/255 for x in nums]
            return (r, g, b)
        except:
            print(f"⚠️ Could not parse RGB color: {value}")
            return None

    # Hex or named colors work natively
    return value


def fetch_image(url_or_path):
    """Load image from URL or local path, returns PIL Image or None."""
    if pd.isna(url_or_path) or not url_or_path:
        return None
    try:
        if hasattr(url_or_path, "read"):  # Flask FileStorage object
            img = Image.open(url_or_path)
        elif str(url_or_path).startswith("http"):
            response = requests.get(url_or_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(url_or_path)
        return img.convert("RGBA")
    except Exception as e:
        print(f"⚠️ Could not load image from {url_or_path}: {e}")
        return None


def combine_images_side_by_side(images, size=(45, 45)):
    """Combines up to 3 PIL images side by side (resized)."""
    images = [img for img in images if img is not None]
    if not images:
        return None
    resized = [img.resize(size) for img in images]
    total_width = size[0] * len(resized)
    combined = Image.new("RGBA", (total_width, size[1]), (255, 255, 255, 0))

    for idx, img in enumerate(resized):
        combined.paste(img, (idx * size[0], 0))
    return combined


# =========================
# Main Plot Function
# =========================

def plot_pie_with_online_images(excel_path, image_scale=0.75, center_image=None, background_image=None, background_alpha=0.3, title="", show=True):
    """
    Draws a pie chart from Excel data.
    Each slice can have up to 3 images, and an optional image can be placed at the center.

    Args:
        excel_path (str): Path to Excel file.
        image_scale (float): Scale factor for images around pie.
        center_image (str): Optional URL or file path for a central image.
        :param background_image:
    """
    # Load Excel
    df = pd.read_excel(excel_path)
    required_cols = ["Name", "Value"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel file must contain column '{col}'")

    df = df.dropna(subset=["Name", "Value"])
    labels = df["Name"].tolist()
    sizes = df["Value"].tolist()


    colors = []
    try:
        with open('mon_to_color.json', 'r', encoding='utf-8') as file:
            poke_map = json.load(file)
        for label in labels:
            colors.append(parse_color(poke_map[label]))
    except FileNotFoundError:
        print("No color mapping found?!")
        colors = [None] * len(labels)
    except KeyError:
        print("Pokemon " + label + " has no mapping")
        colors = [None] * len(labels)

    total_players = sum(sizes)
    percentages = [100 * v / total_players for v in sizes]

    fig, ax = plt.subplots(figsize=(9, 9))

    if background_image:
        bg_img = fetch_image(background_image)
        if bg_img:
            bg_img = bg_img.resize((int(fig.bbox.bounds[2]), int(fig.bbox.bounds[3])))
            bg_img_np = np.array(bg_img)
            fig.figimage(bg_img_np, xo=0, yo=0, alpha=background_alpha, zorder=-1)
        else:
            print("⚠️ Background image could not be loaded.")

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.5),
        normalize=True
    )

    # Plot slice images and percentage text
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x_img = 0.7 * np.cos(np.deg2rad(angle))
        y_img = 0.7 * np.sin(np.deg2rad(angle))
        x_text = 1.2 * np.cos(np.deg2rad(angle))
        y_text = 1.2 * np.sin(np.deg2rad(angle))

        mons = labels[i].split(" ")
        imgs = []
        mega = False
        paldean = False
        galar = False
        paradox_options = ["Iron", "Great", "Scream", "Brute", "Flutter", "Slither", "Sandy", "Roaring", "Walking", "Gouging", "Raging"]
        for j in range(len(mons)):
            if mons[j] == "Mega":
                if mons[j+1] == "Box":
                    imgs.append(fetch_image(generate_image_url(pokemon="Audino", mega=True)))
                    imgs.append(fetch_image(generate_image_url(pokemon="Absol", mega=True)))
                else:
                    mega = True
            elif mons[j] == "Paldean":
                paldean = True
            elif mons[j] == "Galarian":
                galar = True
            elif mons[j] in paradox_options:
                mons[j+1] = mons[j] + "-" + mons[j+1]
            elif mons[j] == "Tera":
                imgs.append(fetch_image(generate_image_url(pokemon="Ogerpon")))
                imgs.append(fetch_image(generate_image_url(pokemon="Noctowl")))
            elif mons[j] == "Tord" and mons[j+1] == "Box":
                imgs.append(fetch_image(generate_image_url(pokemon="Absol", mega=True)))
                imgs.append(fetch_image(generate_image_url(pokemon="Kangaskhan", mega=True)))
            elif mons[j] == "Zacian":
                imgs.append(fetch_image(generate_image_url(pokemon="Zacian", crown=True)))
            elif not mons[j].endswith("'s") and not mons[j] == "Box" and not mons[j] == "Team":
                imgs.append(fetch_image(generate_image_url(pokemon=mons[j], mega=mega, paldean=paldean, galar=galar)))
                mega = False




        combined_img = combine_images_side_by_side(imgs)
        if combined_img:
            imagebox = OffsetImage(combined_img, zoom=image_scale)
            ab = AnnotationBbox(imagebox, (x_img, y_img), frameon=False)
            ax.add_artist(ab)

        # Add percentage text label (outside the chart)
        percent_label = f"{percentages[i]:.1f}%" + " (" + str(sizes[i]) + ")"
        ax.text(
            x_text,
            y_text,
            percent_label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                boxstyle="round,pad=0.3",
                linewidth=1.2
            )
        )

    # Add central image (if provided)
    if center_image:
        center_img = fetch_image(center_image)
        if center_img:
            center_scale = image_scale * 0.80  # 50% smaller
            imagebox = OffsetImage(center_img, zoom=center_scale)
            ab_center = AnnotationBbox(imagebox, (0, 0), frameon=False)
            ax.add_artist(ab_center)
        else:
            print("⚠️ Center image could not be loaded.")

    # Add legend

    ax.legend(
        wedges, labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5)
    )

    # Add total players text below the chart
    plt.figtext(0.5, 0.05, f"Total: {total_players} players", ha="center", fontsize=13, fontweight="bold")
    ax.set(aspect="equal")
    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.95)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if show:
        plt.show()
    return fig


# =========================
# Example Usage
# =========================
app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>Pie Chart Generator</title></head>
<body style="font-family:Arial;text-align:center;">
<h2>📊 Upload Excel File</h2>
<p>Excel must have columns: <b>Name</b> and <b>Value</b></p>
<form method="post" enctype="multipart/form-data">
  <label>Excel File:</label><br>
  <input type="file" name="file" required><br><br>
  <label>Title:</label><br>
  <input type="text" name="title" placeholder="Chart Title"><br><br>
  <label>Center Image (URL or upload):</label><br>
  <input type="text" name="center_url" placeholder="https://..."><br>
  <input type="file" name="center_file"><br><br>
  <label>Background Image (URL or upload):</label><br>
  <input type="text" name="bg_url" placeholder="https://..."><br>
  <input type="file" name="bg_file"><br><br>
  <button type="submit">Generate Chart</button>
</form>
</body>
</html>
"""
HTML_RESULT = """
<!DOCTYPE html>
<html>
<head><title>Chart Ready</title></head>
<body style="font-family:Arial;text-align:center;">
<h2>✅ Your chart is ready!</h2>
<p><a href="{{ download_url }}" download><button>⬇️ Download PNG</button></a></p>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        excel_path = request.files["file"]
        title = request.form.get("title", "Pie Chart")

        # center image: prefer upload > URL
        center_file = request.files.get("center_file")
        center_url = request.form.get("center_url")
        center_image = center_file if center_file and center_file.filename else center_url

        # background image: prefer upload > URL
        bg_file = request.files.get("bg_file")
        bg_url = request.form.get("bg_url")
        background_image = bg_file if bg_file and bg_file.filename else bg_url

        fig = plot_pie_with_online_images(
            excel_path=excel_path,
            center_image=center_image,
            background_image=background_image,
            title=title,
            show=False
        )

        # Save figure to memory and return as image
        img_io = io.BytesIO()
        fig.savefig(img_io, format="png", bbox_inches="tight")
        img_io.seek(0)
        plt.close(fig)
        return send_file(img_io, mimetype="image/png")

    return render_template_string(HTML_FORM)


if __name__ == "__main__":
    app.run()

@app.route("/download/<filename>")
def download_chart(filename):
    return send_file(f"generated/{filename}", as_attachment=True, download_name="chart.png")
