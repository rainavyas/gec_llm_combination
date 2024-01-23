from src.inference.ensemble import MaxvoteEnsModel, MBREnsModel
from .hf_gec import HFGECModel
from src.tools.tools import get_default_device

MODEL_PATHS = {
    'gector-roberta' : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/roberta_1_gectorv2.th',
    'gector-bert'    : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/bert_0_gectorv2.th',
    'gector-xlnet'   : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/xlnet_0_gectorv2.th'
}

def select_model(args):
    mnames = args.model_name
    if len(mnames) == 1:
        args.model_name = mnames[0]
        return _select_single_model(args)
    else:
        models = []
        for curr_mname in mnames:
            args.model_name = curr_mname
            models.append(_select_single_model(args))
        
        if args.ens_type == 'mbr':
            return MBREnsModel(models)
        elif args.ens_type == 'maxvote':
            return MaxvoteEnsModel(models)

def _select_single_model(args):
    if 'gector' in args.model_name:
        mname = args.model_name
        model_path = MODEL_PATHS[mname]
        from src.inference.gector import GectorModel
        transformer_model = mname.split('-')[-1]
        return GectorModel(args, transformer_model=transformer_model, model_path=model_path)
    else:
        device=get_default_device()
        return HFGECModel(device, model_name=args.model_name, data_name=args.data_name)
    
