import argparse
import os
import pathlib
from copy import deepcopy

import tensorflow as tf
import numpy as np

from pretrain import run
from stream_train_test import stream_run


def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    params["optimizer"] = "Adam"
    if args.data_cache_path != "None":
        try:
            pathlib.Path(args.data_cache_path).mkdir(parents=True)
        except:
            print 'exist'
    if args.mode == "pretrain":
        if args.method == "Pretrain":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "baseline_prtrain"
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "dfm_prtrain"
            params["model"] = "MLP_EXP_DELAY"
        elif args.method == "FSIW":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = args.fsiw_pretraining_type+"_cd_"+str(args.CD)
            params["model"] = "MLP_FSIW"
        elif args.method == "ES-DFM":
            params["loss"] = "tn_dp_pretraining_loss"
            params["dataset"] = "tn_dp_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_tn_dp"
        elif args.method == "3class":
            params["loss"] = "softmax_cross_entropy_loss"
            params["dataset"] = "3class_cut_hour_"+str(args.C)
            params["model"] = "MLP_3class"
        elif args.method == "win_time":
            params["loss"] = "win_time_loss"
            params["dataset"] = "likeli_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_likeli"
        elif args.method == "win_time_test":
            params["loss"] = "win_time_loss_test" #"cross_entropy_loss_00"
            #params["loss"] = "cross_entropy_loss_00"
            params["dataset"] = "likeli_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_SIG"
        elif args.method == "win_time_sep":
            params["loss"] = "win_time_loss"
            params["dataset"] = "likeli_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_wintime_sep"
        elif args.method == "win_adapt":
            params["loss"] = "win_select_loss"
            params["dataset"] = "win_adapt_cut_hour_"+str(args.C)
            params["model"] = "MLP_winadapt"
        else:
            raise ValueError(
                "{} method do not need pretraining other than Pretrain".format(args.method))
    else:
        if args.method == "Pretrain":
            params["loss"] = "none_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "Oracle":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "last_30_train_test_dfm"
        elif args.method == "FSIW":
            params["loss"] = "fsiw_loss"
            params["dataset"] = "last_30_train_test_fsiw"
        elif args.method == "ES-DFM":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)
        elif args.method == "ES-DFM10":
            params["loss"] = "esdfm_loss10"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_"+str(args.C)
        elif args.method == "ES-DFM-normal":
            params["loss"] = "esdfm_loss_win" #"esdfm_loss_normal"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)
        elif args.method == "ES-DFM-win":
            params["loss"] = "esdfm_loss_normal_test" #"esdfm_loss_win"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)
        elif args.method == "ES-DFM-FULL":
            params["loss"] = "esdfm_loss_full"
            params["dataset"] = "last_30_train_test_delay_win_time_cut_hour_"+str(args.C)
            #params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
            #    str(args.C)
        elif args.method == "ES-DFM-wines":
            params["loss"] = "esdfm_loss_wines" #"esdfm_loss_win"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)

        elif args.method == "Vanilla":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw_cut_hour_"+str(args.C)
            #params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + +str(args.C)
                
        elif args.method == "FNW":
            #params["loss"] = "fnw_test_loss"
            params["loss"] = "fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNW10":
            params["loss"] = "fnw10_test_loss"
            params["dataset"] = "last_30_train_test_fnw"

        elif args.method == "FNC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNC10":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw"

        elif args.method == "3class":
            params["loss"] = "delay_3class_loss"
            params["dataset"] = "last_30_train_test_3class_cut_hour_"+str(args.C)
            params["model"] = "MLP_3class"
        elif args.method == "likeli":
            params["loss"] = "delay_likelihood_loss"
            params["dataset"] = "last_30_train_test_3class_cut_hour_"+str(args.C)
            params["model"] = "MLP_likeli"
        elif args.method == "delay_win_time":
            params["loss"] = "delay_win_time_loss"
            params["dataset"] = "last_30_train_test_delay_win_time_cut_hour_"+str(args.C)
            params["model"] = "MLP_likeli"
        elif args.method == "delay_win_time_iw":
            params["loss"] = "delay_win_time_iwloss"
            params["dataset"] = "last_30_train_test_delay_win_time_cut_hour_"+str(args.C)
            params["model"] = "MLP_likeli"

        elif args.method == "delay_win_time_sep":
            params["loss"] = "delay_win_time_loss"
            params["dataset"] = "last_30_train_test_delay_win_time_cut_hour_"+str(args.C)
            params["model"] = "MLP_wintime_sep"
        elif args.method == "delay_win_adapt":
            params["loss"] = "delay_win_select_loss"
            params["dataset"] = "last_30_train_test_delay_win_adapt_cut_hour_"+str(args.C)
            params["model"] = "MLP_likeli"
        elif args.method == "delay_win_select":
            params["loss"] = "esdfm_loss_win_weight" #"esdfm_loss" #"delay_win_select_loss"
            params["dataset"] = "last_30_train_test_delay_win_adapt_cut_hour_"+str(args.C)
            #params["model"] = "MLP_winadapt"
        elif args.method == "test":
            params["loss"] = "delay_win_select_loss"
            #params["loss"] = "delay_win_time_loss"
            #params["dataset"] = "last_30_train_test_fnw"
            params["dataset"] = "last_30_train_test_delay_win_adapt_cut_hour_"+str(args.C)
            #params["dataset"] = "last_30_train_test_delay_win_time_cut_hour_"+str(args.C)
            params["model"] = "MLP_winadapt"
            #params["model"] = "MLP_likeli"

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "DFM",
                                 "ES-DFM",
                                 "ES-DFM-win",
                                 "ES-DFM-normal",
                                 "ES-DFM10",
				"ES-DFM-wines",
				"ES-DFM-FULL",
                                 "FNW",
				"FNW10",
                                 "FNC",
                       		"FNC10",
                                 "Pretrain",
                                 "Oracle",
                                 "Vanilla",
				"3class",
  				"likeli",
				"win_time",
				"win_time_test",
  				"win_adapt",
				"delay_win_time",
 				"delay_win_time_iw",
 				"delay_win_adapt",
  				"delay_win_select",
 				"delay_win_time_sep",
  				"win_time_sep",
				"test"
],
                        type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["pretrain", "stream"], help="training mode", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--CD", type=int, default=7,
                        help="counterfactual deadline in FSIW")
    parser.add_argument("--C", type=int, default=0.25,
                        help="elapsed time in ES-DFM")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, required=True,
                        help="path of the data.txt in criteo dataset, e.g. /home/xxx/data.txt")
    parser.add_argument("--data_cache_path", type=str, default="None")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    parser.add_argument("--pretrain_fsiw0_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw0 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_fsiw1_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw1 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_baseline_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained baseline model(Pretrain),  \
                        necessary for the streaming evaluation of \
                            FSIW, ES-DFM, FNW, FNC, Pretrain, Oracle, Vanilla method")
    parser.add_argument("--pretrain_dfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained DFM model,  \
                        necessary for the streaming evaluation of \
                            DFM method")
    parser.add_argument("--pretrain_esdfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained ES-DFM model,  \
                        necessary for the streaming evaluation of \
                        ES-DFM method") 
    parser.add_argument("--pretrain_3class_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained 3class model,  \
                        necessary for the streaming evaluation of \
                        3class method")
    parser.add_argument("--pretrain_wintime_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained wintime model,  \
                        necessary for the streaming evaluation of \
                        wintime method")
    parser.add_argument("--pretrain_sepwintime_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained wintime model,  \
                        necessary for the streaming evaluation of \
                        wintime method")
    parser.add_argument("--pretrain_winselect_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained wintime model,  \
                        necessary for the streaming evaluation of \
                        wintime method")

    parser.add_argument("--fsiw_pretraining_type", choices=["fsiw0", "fsiw1"], type=str, default="None",
                        help="FSIW needs two pretrained weighting model")
    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=1e-6,
                        help="l2 regularizer strength")
    parser.add_argument("--pre_start", type=float, default=0,
                        help="pre-start")
    parser.add_argument("--pre_mid", type=float, default=30,
                        help="pre-mid")
    parser.add_argument("--pre_end", type=float, default=60,
                        help="pre-end")
    parser.add_argument("--start", type=float, default=0,
                        help="start")
    parser.add_argument("--mid", type=float, default=30,
                        help="mid")
    parser.add_argument("--end", type=float, default=60,
                        help="end")
    parser.add_argument("--ddline", type=int, default=-1,
                        help="dd")

    parser.add_argument("--subloss", type=int, default=1,
                        help="--subloss")
    parser.add_argument("--mul", type=float, default=1,
                        help="--subloss")
    parser.add_argument("--win1", type=float, default=0.25,
			help="--subloss")
    parser.add_argument("--win2", type=float, default=0.5,
   			help="--subloss")
    parser.add_argument("--win3", type=float, default=1,
   			help="--subloss")
    parser.add_argument("--delfu", type=int, default=0,
                        help="--subloss")
    parser.add_argument("--pre_trainstart", type=int, default=0,
                        help="--pre_trainstart")
    parser.add_argument("--pre_trainend", type=int, default=30,
                        help="--pre_trainend")
    parser.add_argument("--pre_teststart", type=int, default=30,
                        help="--pre_teststart")
    parser.add_argument("--pre_testend", type=int, default=60,
                        help="--pre_testend")
    parser.add_argument("--cv_mask_day", type=int, default=1,
                        help="--cv_mask_day")
    parser.add_argument("--rnwin", type=int, default=24,
                        help="--cv_mask_day")
    #rnwin
    args = parser.parse_args()
    params = run_params(args)
    tf.random.set_seed(args.seed)
    #tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print("params {}".format(params))
    if args.mode == "pretrain":
        run(params)
    else:
        stream_run(params)
