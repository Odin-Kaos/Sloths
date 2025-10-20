from sloths.module import prepare_data

def test_prepare_data_output():
    train_loader, val_loader, class_names, class_dist, widths, heights = prepare_data()
    assert hasattr(train_loader, "__iter__")
    assert hasattr(val_loader, "__iter__")
    assert isinstance(class_names, list)
    assert not class_dist.empty
