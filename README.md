# Demo rgbd object classifier
running model:  fully connected to fuse views in a ring, average voting (no attention), square ring setting:

    class_score = test_score
    
    class_score = class_score.reshape(-1,N_RING,20)
    
    arg_mean = np.mean(class_score,1)
    
    preds = np.argmax(arg_mean,1)
    

# to run:
- install xvbf (x virtual frame buffer) (if use a headless linux instance)

- trimesh 

- tf & pytorch.


# files:
app.py: main app server

classìfy model: input 26 imgs, output class scores.

extract_feature: extract resnet-50 2048-d for each imgs.

render_26: render 26 colored view around the objects.

# screenshot


![Alt text](pic/demo.png?raw=true "screenshot")
