# def train_mlp_model(args, dataloader, valid_loader):
#     print("creating MLP model...")
#     num_gpus = torch.cuda.device_count()
#     args.num_workers = args.num_workers * num_gpus
    
#     flame = FLAME(args)

#     # init wandb log
#     if args.wandb_log:
#         wandb.init(
#             project="face_expression",
#             name=args.arch,
#             config=args,
#             settings=wandb.Settings(start_method="fork"),
#             dir="./wandb"
#         )
    
#     model = PureMLP(
#         args.latent_dim,
#         args.input_motion_length,
#         args.layers,
#         args.sparse_dim,
#         args.motion_nfeat,
#     )

#     if num_gpus > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         dist_util.setup_dist()
#         model = torch.nn.DataParallel(model).cuda()
#         print(
#             "Total params: %.2fM"
#             % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
#         )
#     else:
#         dist_util.setup_dist(args.device)
#         model.to(dist_util.dev())
#         print(
#             "Total params: %.2fM"
#             % (sum(p.numel() for p in model.parameters()) / 1000000.0)
#         )

#     # initialize optimizer
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=args.lr, weight_decay=args.weight_decay
#     )
#     nb_iter = 0
#     total_steps = len(dataloader) * args.num_epoch
#     log_gt_videos_flag = True
#     for epoch in tqdm(range(args.num_epoch)):
#         model.train()
#         for (motion_target, motion_input, shape) in tqdm(dataloader):

#             loss_dict, optimizer, current_lr = train_step(
#                 motion_input,
#                 motion_target,
#                 model,
#                 optimizer,
#                 nb_iter,
#                 total_steps,
#                 args.lr,
#                 args.lr / 10.0,
#                 dist_util.dev(),
#                 args.lr_anneal_steps,
#             )

            
#             loss_dict["lr"] = current_lr
#             loss_dict["epoch"] = epoch
#             if args.wandb_log:
#                 wandb_log_dict = {}
#                 for k, v in loss_dict.items():
#                     wandb_log_dict["train/"+k] = v
#                 wandb.log(wandb_log_dict)
#             nb_iter += 1
            
#         if (epoch + 1) % args.log_interval == 0:
#             model.eval()
#             total_expr_loss = 0.0
#             dist_verts, dist_lmk3d = 0.0, 0.0
#             eval_steps = 0.0
#             log_gt_videos, log_rec_videos = [], []
#             with torch.no_grad():
#                 print(f"eval epoch {epoch} ...")
#                 for idx in tqdm(range(len(valid_loader))):
#                     motion_target, motion_input, shape, _ = valid_loader[idx]
#                     eval_steps += 1
#                     render_video = True if idx == 0 else False
#                     loss_dict, gt_video_frames, rec_video_frames = val_step(
#                         motion_input,
#                         motion_target,
#                         shape,
#                         model,
#                         flame,
#                         dist_util.dev(),
#                         args.input_motion_length,
#                         render_video
#                     )
#                     if render_video:
#                         log_gt_videos.append(gt_video_frames)
#                         log_rec_videos.append(rec_video_frames)
#                     total_expr_loss += loss_dict["expr_loss"]
#                     dist_verts += loss_dict["verts"]
#                     dist_lmk3d += loss_dict["lmk3d"]
#                 if args.wandb_log:   
#                     log_gt_videos = np.stack(log_gt_videos)
#                     log_rec_videos = np.stack(log_rec_videos)
#                     if log_gt_videos_flag:
#                         wandb.log({
#                             "validation/expr_loss": total_expr_loss / eval_steps,
#                             "validation/verts_dist": dist_verts / eval_steps,
#                             "validation/lmk_3d_68_dist": dist_lmk3d / eval_steps,
#                             "validation/gt_videos": wandb.Video(log_gt_videos, fps=60),
#                             "validation/rec_videos": wandb.Video(log_rec_videos, fps=60),
#                             "validation/epoch": epoch
#                         })
#                         log_gt_videos_flag = False
#                     else:
#                         wandb.log({
#                             "validation/expr_loss": total_expr_loss / eval_steps,
#                             "validation/verts_dist": dist_verts / eval_steps,
#                             "validation/lmk_3d_68_dist": dist_lmk3d / eval_steps,
#                             "validation/rec_videos": wandb.Video(log_rec_videos, fps=60),
#                             "validation/epoch": epoch
#                         })
#                 print(loss_dict)
                        
#         if (epoch + 1) % args.save_interval == 0:
#             with open(
#                 os.path.join(args.save_dir, "model-epoch-" + str(epoch) + "-step-" + str(nb_iter) + ".pth"),
#                 "wb",
#             ) as f:
#                 torch.save(model.state_dict(), f)