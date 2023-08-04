from option import get_opt
from models.TrainModel import lumos


if __name__ == "__main__":
    opt = get_opt()
    opt.test = True
    model = lumos(opt)
    model.load_ckpt()
    dataset_test = Dataset(opt, stage='test')
    dataloader_test = create_dataset(opt, dataset_test, shuffle=False, val=True)
    for i, data in enumerate(dataloader_test):
        model.set_input(data)
        model.eval()
        model.test()
    model.print_metric()