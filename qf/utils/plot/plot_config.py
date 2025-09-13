from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class MatplotlibConfig:
    """Configuration class for matplotlib-specific settings."""

    # Figure settings
    figsize: Tuple[float, float] = (7.5, 3)
    dpi: int = 300

    # Line and style settings
    linewidth: float = 1.5
    alpha: float = 0.7
    grid: bool = True
    grid_alpha: float = 0.5
    grid_linestyle: str = "--"

    # Color settings
    colormap: str = "tab10"
    colormap_type: Literal["qualitative", "continuous"] = "qualitative"

    # Common qualitative colormaps
    QUALITATIVE_COLORMAPS = [
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
        "Set1",
        "Set2",
        "Set3",
        "Paired",
        "Accent",
        "Dark2",
        "Pastel1",
        "Pastel2",
    ]

    # Common continuous colormaps
    CONTINUOUS_COLORMAPS = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "coolwarm",
        "RdBu",
        "RdYlBu",
        "Spectral",
        "Blues",
        "Greens",
        "Reds",
        "Oranges",
        "Purples",
    ]

    def get_colormap_type(self) -> Literal["qualitative", "continuous"]:
        """Automatically detect colormap type based on the colormap name."""
        if self.colormap_type != "qualitative":  # If explicitly set to continuous
            return self.colormap_type

        # Auto-detect based on colormap name
        if self.colormap in self.QUALITATIVE_COLORMAPS:
            return "qualitative"
        elif self.colormap in self.CONTINUOUS_COLORMAPS:
            return "continuous"
        else:
            # Default to qualitative for unknown colormaps
            return "qualitative"

    # Font settings
    font_size: int = 9
    axes_titlesize: int = 9
    axes_labelsize: int = 8
    xtick_labelsize: int = 8
    ytick_labelsize: int = 8
    legend_fontsize: int = 8

    # PGF/LaTeX settings
    pgf_enabled: bool = True
    pgf_texsystem: str = "pdflatex"
    text_usetex: bool = True
    pgf_rcfonts: bool = False
    axes_formatter_use_mathtext: bool = True

    def get_rc_params(self) -> Dict[str, Any]:
        """Get matplotlib rcParams dictionary."""
        return {
            "font.size": self.font_size,
            "axes.titlesize": self.axes_titlesize,
            "axes.labelsize": self.axes_labelsize,
            "xtick.labelsize": self.xtick_labelsize,
            "ytick.labelsize": self.ytick_labelsize,
            "legend.fontsize": self.legend_fontsize,
            "axes.grid": self.grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": self.grid_linestyle,
        }

    def get_pgf_rc_params(self) -> Dict[str, Any]:
        """Get PGF-specific rcParams dictionary."""
        return {
            "pgf.texsystem": self.pgf_texsystem,
            "text.usetex": self.text_usetex,
            "pgf.rcfonts": self.pgf_rcfonts,
            "axes.formatter.use_mathtext": self.axes_formatter_use_mathtext,
        }

    @classmethod
    def jupyter(cls) -> "MatplotlibConfig":
        """Create a Jupyter-optimized MatplotlibConfig."""
        return cls(
            # Figure settings optimized for screen display
            figsize=(7.5, 3),  # Larger for better visibility
            dpi=150,  # Higher DPI for crisp display on screens
            # Font settings optimized for screen reading
            # font_size=12,
            # axes_titlesize=14,
            # axes_labelsize=12,
            # xtick_labelsize=10,
            # ytick_labelsize=10,
            # legend_fontsize=10,
            # Line and style settings
            # linewidth=2.0,  # Thicker lines for better visibility
            # alpha=0.8,
            # grid=True,
            # grid_alpha=0.3,
            # grid_linestyle="-",
            # Color settings optimized for screen display
            # colormap="tab10",
            # colormap_type="qualitative",
            # Jupyter-specific settings
            pgf_enabled=False,  # Disable PGF for interactive display
            # text_usetex=False,  # Disable LaTeX for faster rendering
            # axes_formatter_use_mathtext=False,
        )

    def get_jupyter_rc_params(self) -> Dict[str, Any]:
        """Get matplotlib rcParams dictionary optimized for Jupyter."""
        return {
            "font.size": self.font_size,
            "axes.titlesize": self.axes_titlesize,
            "axes.labelsize": self.axes_labelsize,
            "xtick.labelsize": self.xtick_labelsize,
            "ytick.labelsize": self.ytick_labelsize,
            "legend.fontsize": self.legend_fontsize,
            "axes.grid": self.grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": self.grid_linestyle,
            "figure.dpi": self.dpi,
            "text.usetex": self.text_usetex,
            "axes.formatter.use_mathtext": self.axes_formatter_use_mathtext,
            # Jupyter-specific optimizations
            "figure.autolayout": True,  # Better layout in notebook cells
            "figure.constrained_layout.use": True,  # Use constrained layout
            "savefig.bbox": "tight",  # Tight bounding box for better fit
            "savefig.pad_inches": 0.1,  # Small padding
        }


@dataclass
class LinePlotConfig:
    """Configuration specifically for line plots."""

    # Line settings
    linestyles: bool = True
    smoothing: Optional[int] = None
    legend_loc: Literal["outside right", "outside top"] = "outside right"
    max_xticks: int = 12

    # Axis settings
    y_limits: Optional[Tuple[float, float]] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None


@dataclass
class ConfidenceIntervalConfig:
    """Configuration specifically for confidence interval plots."""

    sigma: int = 3
    fill_alpha: float = 0.3
    legend_loc: Literal["outside right", "outside top"] = "outside right"


@dataclass
class DualAxisConfig:
    """Configuration specifically for dual-axis plots."""

    num_yticks: int = 5
    round_base: int = 100
    verbosity: int = 0
    max_xticks: int = 12
    max_entries: Optional[int] = None
    y_limits_left: Optional[Tuple[float, float]] = None
    y_limits_right: Optional[Tuple[float, float]] = None


@dataclass
class GridPlotConfig:
    """Configuration specifically for grid plots."""

    n_cols: int = 3
    bins: int = 50
    log_y_scale: bool = False
    ylim: Optional[Tuple[float, float]] = None
    hist_color: Optional[str] = None
    kde_color: Optional[str] = None


@dataclass
class SaveConfig:
    """Configuration for saving plots."""

    save_dir: str = "plots"
    save_formats: List[str] = field(default_factory=lambda: ["pgf", "png"])
    # bbox_inches: str = "tight"


@dataclass
class PlotConfig:
    """Main configuration class that combines all plot configurations."""

    # Core configurations
    matplotlib: MatplotlibConfig = field(default_factory=MatplotlibConfig)
    save: SaveConfig = field(default_factory=SaveConfig)

    # Plot-specific configurations
    line_plot: LinePlotConfig = field(default_factory=LinePlotConfig)
    confidence_interval: ConfidenceIntervalConfig = field(
        default_factory=ConfidenceIntervalConfig
    )
    dual_axis: DualAxisConfig = field(default_factory=DualAxisConfig)
    grid_plot: GridPlotConfig = field(default_factory=GridPlotConfig)

    # Legacy support - these will be deprecated but kept for backward compatibility
    figsize: Tuple[float, float] = field(init=False)
    linewidth: float = field(init=False)
    grid: bool = field(init=False)
    colormap: str = field(init=False)
    alpha: float = field(init=False)
    dpi: int = field(init=False)
    pgf_enabled: bool = field(init=False)
    save_formats: List[str] = field(init=False)
    save_dir: str = field(init=False)
    sigma: int = field(init=False)
    n_cols: int = field(init=False)
    num_yticks: int = field(init=False)
    round_base: int = field(init=False)
    verbosity: int = field(init=False)

    def __post_init__(self):
        """Initialize legacy properties for backward compatibility."""
        self.figsize = self.matplotlib.figsize
        self.linewidth = self.matplotlib.linewidth
        self.grid = self.matplotlib.grid
        self.colormap = self.matplotlib.colormap
        self.alpha = self.matplotlib.alpha
        self.dpi = self.matplotlib.dpi
        self.pgf_enabled = self.matplotlib.pgf_enabled
        self.save_formats = self.save.save_formats
        self.save_dir = self.save.save_dir
        self.sigma = self.confidence_interval.sigma
        self.n_cols = self.grid_plot.n_cols
        self.num_yticks = self.dual_axis.num_yticks
        self.round_base = self.dual_axis.round_base
        self.verbosity = self.dual_axis.verbosity
        self.legend_loc = self.line_plot.legend_loc

    def asdict(self):
        """Convert configuration to dictionary."""
        return {
            "matplotlib": self.matplotlib.__dict__,
            "save": self.save.__dict__,
            "line_plot": self.line_plot.__dict__,
            "confidence_interval": self.confidence_interval.__dict__,
            "dual_axis": self.dual_axis.__dict__,
            "grid_plot": self.grid_plot.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PlotConfig":
        """Create PlotConfig from dictionary."""
        matplotlib_config = MatplotlibConfig(**config_dict.get("matplotlib", {}))
        save_config = SaveConfig(**config_dict.get("save", {}))
        line_plot_config = LinePlotConfig(**config_dict.get("line_plot", {}))
        confidence_interval_config = ConfidenceIntervalConfig(
            **config_dict.get("confidence_interval", {})
        )
        dual_axis_config = DualAxisConfig(**config_dict.get("dual_axis", {}))
        grid_plot_config = GridPlotConfig(**config_dict.get("grid_plot", {}))

        return cls(
            matplotlib=matplotlib_config,
            save=save_config,
            line_plot=line_plot_config,
            confidence_interval=confidence_interval_config,
            dual_axis=dual_axis_config,
            grid_plot=grid_plot_config,
        )

    @classmethod
    def slim(cls) -> "PlotConfig":
        """Create a slim PlotConfig with only figsize modified."""
        return cls(
            matplotlib=MatplotlibConfig(figsize=(7.5, 1.5)),
        )

    @classmethod
    def jupyter(cls) -> "PlotConfig":
        """Create a Jupyter notebook optimized PlotConfig."""
        return cls(
            matplotlib=MatplotlibConfig.jupyter(),  # Use Jupyter-optimized MatplotlibConfig
            save=SaveConfig(
                save_formats=["png", "svg", "pgf"]
            ),  # Optimized formats for web display + PGF for LaTeX
        )

    @staticmethod
    def jupyter_config() -> MatplotlibConfig:
        """Get a Jupyter-optimized configuration."""
        return MatplotlibConfig.jupyter()
