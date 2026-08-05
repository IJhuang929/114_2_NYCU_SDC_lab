## Track1
config = {
    'lr_xyz': 0.0002,
    'lr_color': 0.0025,
    'lr_opacity': 0.05,
    'lr_scaling': 0.005,
    'lr_rotation': 0.001,
    'lr_decay': 0.01 ** (1 / 73500),
    'checkpoint_interval': 10,
    'init_scale_lidar': 0.15,    
    'init_scale_sky': 0.3,
    'init_opacity': 0.2,
    'refine_start_iter': 500,  
    'refine_stop_iter': 50000,  
    'refine_every': 200,  
    'checkpoint_path': None, 
    'prune_opa': 0.05,
    'grow_grad2d': 0.0001,
    'grow_scale3d': 0.01,
    'grow_scale2d': 0.05,
    'prune_scale3d': 0.15,
    'prune_scale2d': 0.15,

}

## Track2
config = {
    'lr_xyz': 0.0005,
    'lr_color': 0.0025,
    'lr_opacity': 0.025,
    'lr_scaling': 0.005,
    'lr_rotation': 0.001,
    'lr_decay': 0.99995,
    'checkpoint_interval': 5,
    'init_scale_lidar': 0.1,      # Track3 用 0.05,Track2 點雲較稀,放大到 0.1
    'init_scale_sky': 0.3,
    'init_opacity': 0.2,
    'refine_start_iter': 500,     # Track3 用 1500(因為 init 2.1M 太多);Track2 init 47萬合理,500 即可開始
    'refine_stop_iter': 30000,    # 30 epoch=39k iter,densify 跑前 ~64%
    'refine_every': 200,
    'reset_every': 50000,
    'prune_opa': 0.01,
    'grow_grad2d': 0.0002,
    'grow_scale3d': 0.01,
    'prune_scale3d': 0.1,
    'checkpoint_path': None,
    }

## Track3
config = {
        'lr_xyz': 0.0005,
        'lr_color': 0.0025,
        'lr_opacity': 0.025,
        'lr_scaling': 0.005,
        'lr_rotation': 0.001,
        'lr_decay': 0.99995,
        'checkpoint_interval': 5,
        'init_scale_lidar': 0.05,
        'init_scale_sky': 0.3,
        'init_opacity': 0.2,
        'refine_start_iter': 1500,
        'refine_stop_iter': 20000,    # densify 只跑前 ~40% 訓練量
        'refine_every': 200,
        'reset_every': 50000,         # 50 epoch 內不 reset
        'prune_opa': 0.01,            # 加強剪枝避免冗餘
        'grow_grad2d': 0.0002,        # split 門檻拉嚴,避免 over-densify
        'grow_scale3d': 0.01,         # gsplat 預設
        'prune_scale3d': 0.1,         # 大於 10% 場景的 Gaussian 砍掉
        'checkpoint_path': None,
}