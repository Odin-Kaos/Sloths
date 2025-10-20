from sloths.demo.app_gradio import run_experiment

def test_run_experiment_short():
    data_dir = "data"   # o un dataset mínimo de test
    seed = 1243
    img_size = 64
    batch_size = 4
    epochs = 3
    color_palette = "default"

    results = run_experiment(data_dir, seed, img_size, batch_size, epochs, color_palette)
    message, *figs = results

    assert "Training completed" in message
    assert len(figs) > 0
