"""Tests del módulo de figuras: se genera un PNG no vacío, sin abrir ventana."""

import matplotlib
import pytest

from predictive_maintenance import figures


def test_backend_is_agg():
    # En macOS matplotlib intenta abrir ventanas y cuelga el CLI (CLAUDE.md
    # §14.4); importar `figures` debe haber fijado el backend sin interfaz.
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize(
    "nombre_funcion",
    ["figure_histograms", "figure_boxplots", "figure_correlation", "figure_prevalence"],
)
def test_figure_functions_write_nonempty_png(lab180_df, lab180_spec, tmp_path, nombre_funcion):
    funcion = getattr(figures, nombre_funcion)
    ruta = funcion(lab180_df, lab180_spec, tmp_path / f"{nombre_funcion}.png")
    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_make_eda_figures_returns_four_existing_paths(lab180_df, lab180_spec, tmp_path):
    rutas = figures.make_eda_figures(lab180_df, lab180_spec, tmp_path / "figs")
    assert len(rutas) == 4
    nombres = {ruta.name for ruta in rutas}
    assert nombres == {
        "histogramas_por_clase.png",
        "boxplots_por_clase.png",
        "correlacion.png",
        "prevalencia.png",
    }
    for ruta in rutas:
        assert ruta.exists()
        assert ruta.stat().st_size > 0
