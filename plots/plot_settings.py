import matplotlib.pyplot as plt
from pathlib import Path

import copy

import logging

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)


ROOT = Path(__file__).resolve().parent.parent


if ROOT.name != "overhead-time-multiplexing":
    raise ValueError(
        f"Expected base path to be 'overhead-time-multiplexing', but got '{ROOT.name}'"
    )
RESULTS_DIR = ROOT / "results"
PLOT_DIR = ROOT / "plots" / "pdf"
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_COLUMN_WIDTH = 3.35  # inches
DOUBLE_COLUMN_WIDTH = 7.0  # inches

# based on Wong, B. Color coding. Nat Methods 7, 573 (2010). https://doi.org/10.1038/nmeth0810-573
wong_colors = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#CC79A7",  # Reddish Purple
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#F0E442",  # Yellow
]


plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "cm",  # Computer Modern for math
        # "text.usetex": True,
        # Axes and labels
        "axes.titlesize": 9,  # Panel titles (a), (b), etc.
        "axes.labelsize": 9,  # Axis labels
        "xtick.labelsize": 8,  # Tick labels
        "ytick.labelsize": 8,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        # Lines and markers
        "lines.linewidth": 0.5,  # Minimum 0.18mm = ~0.5 points
        "lines.markersize": 3,  # Minimum 1mm diameter ≈ 3 points
        "lines.markerfacecolor": "None",  # Hollow markers
        "patch.linewidth": 0.5,
        # Figure settings
        "figure.dpi": 300,  # High resolution for publication
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Grid (generally avoid unless necessary)
        "axes.grid": False,
        "grid.color": "#cccccc",  # Light gray
        "grid.linestyle": "--",  # Dashed
        "grid.linewidth": 0.3,  # Very thin
        "grid.alpha": 0.7,  # Semi-transparent
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        # Spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,  # TrueType fonts
    }
)
plt.rcParams.update(
    {
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def convert_fig_to_dark(fig, target_text_color="white", target_bg_color="black"):
    """
    Creates a copy of the figure and converts all text elements and backgrounds to dark theme colors.
    Also removes insets which can have problematic white backgrounds.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The original figure to convert
    target_text_color : str, optional
        Color for text elements (default: 'white')
    target_bg_color : str, optional
        Color for backgrounds (default: 'black')

    Returns:
    --------
    matplotlib.figure.Figure
        A new figure with dark theme applied and insets removed
    """
    # Create a deep copy of the figure
    dark_fig = copy.deepcopy(fig)

    # Set figure background
    dark_fig.patch.set_facecolor(target_bg_color)

    # Process all axes in the figure
    for ax in dark_fig.get_axes():
        _convert_axis_to_dark(ax, target_text_color, target_bg_color)

    # Convert figure-level text (suptitle, etc.)
    for text in dark_fig.texts:
        text.set_color(target_text_color)
        text.set_fontfamily("Lato")

    return dark_fig


def _convert_axis_to_dark(ax, target_text_color, target_bg_color):
    """Convert a single axis to dark theme, removing problematic insets."""

    # Check if this is a small inset (likely from coupling maps)
    pos = ax.get_position()
    is_small_inset = pos.width < 0.4 and pos.height < 0.4

    if is_small_inset:
        # Remove the inset entirely by making it invisible
        ax.set_visible(False)
        print(f"Removed small inset with position: {pos}")
        return  # Skip further processing of this axis

    # For main axes, proceed with normal dark conversion
    ax.set_facecolor(target_bg_color)

    # Convert all text elements
    _convert_text_elements(ax, target_text_color)

    # Convert axis elements
    _convert_axis_elements(ax, target_text_color)

    # Convert line colors
    _convert_line_colors(ax, target_text_color)

    # Convert coupling map colors (for main plots)
    _convert_coupling_map_colors(ax, target_text_color)

    # Convert legend if present
    if ax.get_legend():
        _convert_legend(ax.get_legend(), target_text_color, target_bg_color)

    # Handle child elements
    for artist in ax.get_children():
        if hasattr(artist, "get_texts") and hasattr(artist, "get_frame"):
            _convert_legend(artist, target_text_color, target_bg_color)
        elif isinstance(artist, plt.Axes):
            # Recursively process child axes (which might be insets)
            _convert_axis_to_dark(artist, target_text_color, target_bg_color)


def _convert_text_elements(ax, target_color):
    """Convert all text elements in an axis to the target color and Lato font."""

    # Title and labels
    if ax.get_title():
        ax.title.set_color(target_color)
        ax.title.set_fontfamily("Lato")
    if ax.get_xlabel():
        ax.xaxis.label.set_color(target_color)
        ax.xaxis.label.set_fontfamily("Lato")
    if ax.get_ylabel():
        ax.yaxis.label.set_color(target_color)
        ax.yaxis.label.set_fontfamily("Lato")

    # Tick labels
    for tick in ax.get_xticklabels():
        tick.set_color(target_color)
        tick.set_fontfamily("Lato")
    for tick in ax.get_yticklabels():
        tick.set_color(target_color)
        tick.set_fontfamily("Lato")

    # All other text objects in the axis
    for text in ax.texts:
        if text.get_color() in ["black", "k", (0, 0, 0), (0.0, 0.0, 0.0)]:
            text.set_color(target_color)
        text.set_fontfamily("Lato")


def _convert_axis_elements(ax, target_color):
    """Convert axis spines, ticks, and grid to the target color."""

    # Spines - convert ALL spines to target color (not just black ones)
    for spine in ax.spines.values():
        spine.set_edgecolor(target_color)

    # Tick marks
    ax.tick_params(colors=target_color)

    # Additional axis line elements
    # Note: XAxis/YAxis objects don't have set_color method

    # Axis line properties
    for axis in [ax.xaxis, ax.yaxis]:
        axis.label.set_color(target_color)
        for tick in axis.get_ticklines():
            tick.set_color(target_color)
        for tick in axis.get_minorticklines():
            tick.set_color(target_color)

    # Grid (if visible)
    if ax.get_axisbelow() is not None:  # Grid is present
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            if line.get_color() in ["black", "k", (0, 0, 0), (0.0, 0.0, 0.0)]:
                line.set_color("gray")  # Use gray for grid lines for better visibility


def _convert_line_colors(ax, target_color):
    """Convert black lines and problematic colors to white/better colors for dark theme."""

    import numpy as np
    import matplotlib.colors as mcolors

    # Define color mappings for dark theme - UPDATED with green-yellow and higher alpha
    dark_color_map = {
        # Original color -> Dark theme replacement
        "green": "#ADFF2F",  # Changed to GreenYellow instead of bright lime green
        "blue": "#66B3FF",  # Bright sky blue (lighter, more visible)
        "red": "#FF6666",  # Bright coral red (softer, more visible)
        "black": target_color,  # Convert black to white/target
        "k": target_color,  # Convert 'k' to white/target
        "#000000": target_color,  # Convert hex black to white/target
        # Add more mappings as needed
    }

    def convert_color(color):
        """Convert a single color using the mapping."""
        # Handle string colors
        if isinstance(color, str) and color.lower() in dark_color_map:
            return dark_color_map[color.lower()]

        # Handle RGB tuples for exact matches
        if hasattr(color, "__len__") and len(color) >= 3:
            color_tuple = tuple(color[:3])
            # Check for black in various formats
            if color_tuple in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                return target_color
            # Check for standard RGB values (approximations)
            elif color_tuple in [(0, 1, 0), (0.0, 1.0, 0.0)]:  # Pure green
                return "#ADFF2F"  # Changed to GreenYellow
            elif color_tuple in [(0, 0, 1), (0.0, 0.0, 1.0)]:  # Pure blue
                return "#66B3FF"
            elif color_tuple in [(1, 0, 0), (1.0, 0.0, 0.0)]:  # Pure red
                return "#FF6666"

        return color  # Return unchanged if no mapping found

    def colors_equal(color1, color2):
        """Safely compare colors, handling arrays and different formats."""
        try:
            if isinstance(color1, str) or isinstance(color2, str):
                return color1 == color2

            # Convert to numpy arrays for comparison
            c1 = np.asarray(color1)
            c2 = np.asarray(color2)

            # Handle different shapes
            if c1.shape != c2.shape:
                return False

            return np.allclose(c1, c2, rtol=1e-5)
        except Exception:
            return False

    # Get all line objects from the axis
    for line in ax.get_lines():
        line_color = line.get_color()
        new_color = convert_color(line_color)
        if not colors_equal(new_color, line_color):
            line.set_color(new_color)

    # Handle collections (fill_between, scatter, etc.)
    for collection in ax.collections:
        # Convert face colors
        if hasattr(collection, "get_facecolor") and hasattr(collection, "set_facecolor"):
            face_colors = collection.get_facecolor()
            if hasattr(face_colors, "shape") and len(face_colors.shape) > 1:
                # Multiple face colors
                new_face_colors = face_colors.copy()
                changed = False
                for i, color in enumerate(face_colors):
                    new_color = convert_color(color)
                    if isinstance(new_color, str):
                        # Convert hex color to RGBA
                        new_color = mcolors.to_rgba(new_color)
                    if not colors_equal(new_color, color):
                        new_face_colors[i] = new_color
                        changed = True
                if changed:
                    collection.set_facecolor(new_face_colors)
            else:
                # Single face color
                new_color = convert_color(face_colors)
                if not colors_equal(new_color, face_colors):
                    collection.set_facecolor(new_color)

        # Convert edge colors
        if hasattr(collection, "get_edgecolor") and hasattr(collection, "set_edgecolor"):
            edge_colors = collection.get_edgecolor()
            if hasattr(edge_colors, "shape") and len(edge_colors.shape) > 1:
                # Multiple edge colors
                new_edge_colors = edge_colors.copy()
                changed = False
                for i, color in enumerate(edge_colors):
                    new_color = convert_color(color)
                    if isinstance(new_color, str):
                        # Convert hex color to RGBA
                        new_color = mcolors.to_rgba(new_color)
                    if not colors_equal(new_color, color):
                        new_edge_colors[i] = new_color
                        changed = True
                if changed:
                    collection.set_edgecolor(new_edge_colors)
            else:
                # Single edge color
                new_color = convert_color(edge_colors)
                if not colors_equal(new_color, edge_colors):
                    collection.set_edgecolor(new_color)


def _convert_legend(legend, target_text_color, target_bg_color):
    """Convert legend elements to dark theme with enhanced color mapping."""

    import numpy as np
    import matplotlib.colors as mcolors

    # Define the same color mappings for legend handles - UPDATED
    dark_color_map = {
        "green": "#ADFF2F",  # Changed to GreenYellow
        "blue": "#66B3FF",
        "red": "#FF6666",
        "black": target_text_color,
        "k": target_text_color,
        "#000000": target_text_color,
    }

    def convert_color(color):
        """Convert a single color using the mapping."""
        if isinstance(color, str) and color.lower() in dark_color_map:
            return dark_color_map[color.lower()]

        if hasattr(color, "__len__") and len(color) >= 3:
            color_tuple = tuple(color[:3])
            if color_tuple in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                return target_text_color
            elif color_tuple in [(0, 1, 0), (0.0, 1.0, 0.0)]:
                return "#ADFF2F"  # Changed to GreenYellow
            elif color_tuple in [(0, 0, 1), (0.0, 0.0, 1.0)]:
                return "#66B3FF"
            elif color_tuple in [(1, 0, 0), (1.0, 0.0, 0.0)]:
                return "#FF6666"

        return color

    def colors_equal(color1, color2):
        """Safely compare colors, handling arrays and different formats."""
        try:
            if isinstance(color1, str) or isinstance(color2, str):
                return color1 == color2

            # Convert to numpy arrays for comparison
            c1 = np.asarray(color1)
            c2 = np.asarray(color2)

            # Handle different shapes
            if c1.shape != c2.shape:
                return False

            return np.allclose(c1, c2, rtol=1e-5)
        except Exception:
            return False

    # Legend frame
    legend.get_frame().set_facecolor(target_bg_color)
    legend.get_frame().set_edgecolor(target_text_color)

    # Legend text
    for text in legend.get_texts():
        text.set_color(target_text_color)
        text.set_fontfamily("Lato")

    # Legend title
    if legend.get_title():
        legend.get_title().set_color(target_text_color)
        legend.get_title().set_fontfamily("Lato")

    # Legend handles with enhanced color conversion and INCREASED ALPHA
    for handle in legend.legend_handles:
        # Handle Line2D objects
        if hasattr(handle, "get_color") and hasattr(handle, "set_color"):
            line_color = handle.get_color()
            new_color = convert_color(line_color)
            if not colors_equal(new_color, line_color):
                handle.set_color(new_color)

        # Handle patches (rectangles, circles, polygons, etc.)
        if hasattr(handle, "get_facecolor") and hasattr(handle, "set_facecolor"):
            face_color = handle.get_facecolor()
            new_color = convert_color(face_color)
            if not colors_equal(new_color, face_color):
                # Convert to RGBA and add INCREASED transparency (0.8 instead of 0.2)
                if isinstance(new_color, str):
                    new_color = mcolors.to_rgba(new_color)
                # Set alpha to 0.8 for main colors (red, blue, green)
                if len(new_color) >= 4:
                    new_color = (*new_color[:3], 0.8)  # INCREASED alpha to 0.8
                else:
                    new_color = (*new_color, 0.8)  # INCREASED alpha to 0.8
                handle.set_facecolor(new_color)

        if hasattr(handle, "get_edgecolor") and hasattr(handle, "set_edgecolor"):
            edge_color = handle.get_edgecolor()
            new_color = convert_color(edge_color)
            if not colors_equal(new_color, edge_color):
                # Keep edge colors more opaque for better definition
                if isinstance(new_color, str):
                    new_color = mcolors.to_rgba(new_color)
                # Set alpha for edges (using 0.8 for consistency)
                if len(new_color) >= 4:
                    new_color = (*new_color[:3], 0.8)  # INCREASED edge opacity to 0.8
                else:
                    new_color = (*new_color, 0.8)  # INCREASED edge opacity to 0.8
                handle.set_edgecolor(new_color)

        # Handle collections (scatter plots, etc.)
        if hasattr(handle, "get_edgecolors") and hasattr(handle, "set_edgecolors"):
            edge_colors = handle.get_edgecolors()
            if hasattr(edge_colors, "shape") and len(edge_colors.shape) > 1:
                # Multiple colors
                new_edge_colors = edge_colors.copy()
                changed = False
                for i, color in enumerate(edge_colors):
                    new_color = convert_color(color)
                    if isinstance(new_color, str):
                        new_color = mcolors.to_rgba(new_color)
                    if not colors_equal(new_color, color):
                        # Add INCREASED transparency (0.8 instead of 0.2)
                        if len(new_color) >= 4:
                            new_color = (*new_color[:3], 0.8)  # INCREASED alpha to 0.8
                        else:
                            new_color = (*new_color, 0.8)  # INCREASED alpha to 0.8
                        new_edge_colors[i] = new_color
                        changed = True
                if changed:
                    handle.set_edgecolors(new_edge_colors)
            else:
                # Single color
                new_color = convert_color(edge_colors)
                if not colors_equal(new_color, edge_colors):
                    if isinstance(new_color, str):
                        new_color = mcolors.to_rgba(new_color)
                    # Add INCREASED transparency (0.8 instead of 0.2)
                    if len(new_color) >= 4:
                        new_color = (*new_color[:3], 0.8)  # INCREASED alpha to 0.8
                    else:
                        new_color = (*new_color, 0.8)  # INCREASED alpha to 0.8
                    handle.set_edgecolors(new_color)


def _convert_coupling_map_colors(ax, target_color):
    """Convert coupling map colors for pygraphviz-based insets."""

    # Handle CircleCollection objects (qubits in coupling maps)
    for collection in ax.collections:
        # Convert face colors (qubit fill colors)
        if hasattr(collection, "get_facecolor") and hasattr(collection, "set_facecolor"):
            face_colors = collection.get_facecolor()
            if hasattr(face_colors, "shape") and len(face_colors.shape) > 1:
                # Multiple face colors
                for i, color in enumerate(face_colors):
                    if len(color) >= 3 and tuple(color[:3]) in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                        face_colors[i] = [1.0, 1.0, 1.0, color[3] if len(color) > 3 else 1.0]
                collection.set_facecolor(face_colors)
            elif hasattr(face_colors, "__len__") and len(face_colors) >= 3:
                # Single face color
                if tuple(face_colors[:3]) in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                    collection.set_facecolor(target_color)

        # Convert edge colors (qubit edge colors) - already handled in _convert_line_colors
        # but let's be explicit for coupling maps
        if hasattr(collection, "get_edgecolor") and hasattr(collection, "set_edgecolor"):
            edge_colors = collection.get_edgecolor()
            if hasattr(edge_colors, "shape") and len(edge_colors.shape) > 1:
                # Multiple edge colors
                for i, color in enumerate(edge_colors):
                    if len(color) >= 3 and tuple(color[:3]) in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                        edge_colors[i] = [1.0, 1.0, 1.0, color[3] if len(color) > 3 else 1.0]
                collection.set_edgecolor(edge_colors)
            elif hasattr(edge_colors, "__len__") and len(edge_colors) >= 3:
                # Single edge color
                if tuple(edge_colors[:3]) in [(0, 0, 0), (0.0, 0.0, 0.0)]:
                    collection.set_edgecolor(target_color)

    # Handle AxesImage objects (qiskit coupling map plots render as images)
    import numpy as np

    for child in ax.get_children():
        if hasattr(child, "get_array") and hasattr(child, "set_array"):
            # This is likely an AxesImage from qiskit coupling map
            try:
                image_data = child.get_array()
                if image_data is not None and hasattr(image_data, "shape"):
                    # Create inverted image data for dark theme
                    if len(image_data.shape) == 3:  # RGB or RGBA
                        # Invert the image: black becomes white, white stays white
                        inverted_data = image_data.copy()
                        # Determine if data is in 0-1 range or 0-255 range
                        max_val = image_data.max()
                        if max_val <= 1.0:
                            # Data is in 0-1 range
                            threshold = 0.1
                            white_val = [1.0, 1.0, 1.0]
                        else:
                            # Data is in 0-255 range
                            threshold = 25  # ~10% of 255
                            white_val = [255.0, 255.0, 255.0]

                        # For each pixel, if it's close to black, make it white
                        mask = np.all(
                            image_data[:, :, :3] <= threshold, axis=2
                        )  # Find black pixels
                        if len(image_data.shape) == 3 and image_data.shape[2] == 4:
                            # RGBA
                            inverted_data[mask] = white_val + [image_data[mask, 3].mean()]
                        else:
                            # RGB
                            inverted_data[mask] = white_val
                        child.set_array(inverted_data)
                    elif len(image_data.shape) == 2:  # Grayscale
                        # For grayscale, invert black to white
                        inverted_data = image_data.copy()
                        max_val = image_data.max()
                        if max_val <= 1.0:
                            threshold = 0.1
                            white_val = 1.0
                        else:
                            threshold = 25  # ~10% of 255
                            white_val = 255.0
                        inverted_data[image_data <= threshold] = white_val  # Black becomes white
                        child.set_array(inverted_data)
            except Exception:
                # If image processing fails, skip this child
                pass


def save_fig(fig, name):
    """
    Enhanced save function with proper dark theme conversion.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    name : str
        Base filename (without extension)
    """

    fig.savefig(
        PLOT_DIR / f"{name}.pdf",
        dpi=300,
        # bbox_inches="tight",
        pad_inches=0.02,
    )

    # Create and save dark version
    dark_fig = convert_fig_to_dark(fig)

    # Ensure dark directory exists
    # dark_dir = PLOT_DIR / "dark"
    dark_dir = PLOT_DIR
    dark_dir.mkdir(parents=True, exist_ok=True)

    dark_fig.savefig(
        dark_dir / f"{name}_dark.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="black",  # Ensure the saved figure has black background
        edgecolor="none",
    )

    # Clean up the temporary dark figure
    plt.close(dark_fig)
