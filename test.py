from option import get_opt
from models.TestModel import lumos
from data.TestDataset import create_dataset
# from data.TestDataset_albedos import create_dataset

if __name__ == "__main__":
    opt = get_opt()
    opt.test = True
    model = lumos(opt).cuda()
    model.load_ckpt()
    dataloader_test = create_dataset(opt)
    for i, data in enumerate(dataloader_test):
        model.set_input(data)
        # model.eval()
        # model.forward()
        model.test()
        # model.save_albedos()
        # model.save_inferred()
    # model.print_metric()