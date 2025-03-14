from scipy.io.matlab.tests.test_mio import CN


def add_condseg_config(cfg):
    # cfg.MODEL.CONDSEG_STAGE1 = CN()

    cfg.INPUT.AUG_TYPE = "weak"
    cfg.INPUT.AUG_EXTRA = True
