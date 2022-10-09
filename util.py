import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import torch 

def laplacian_uniform_2d(v, l):
    V = v.shape[0]
    L = l.shape[0]
    
    #neighbor indices 
    ii = l[:,[1,0]].flatten()
    jj = l[:,[0,1]].flatten()
    adj = torch.stack([torch.cat([ii,jj]), torch.cat([jj,ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)    
    diag_idx = adj[0]
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))
    L = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()
    return L

def plot_mesh2d(v, l, y_lim=None, x_lim=None, return_ax=False, showfig=False, filename=None):
    #with sns.axes_style('dark'):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5,5)
    ax.set_aspect('equal', adjustable='box')

    vtx = v[l, :]
    x = vtx[:, :, 0].reshape((-1, 1))
    y = vtx[:, :, 1].reshape((-1, 1))
    ax.plot(x, y, linewidth=4, color='#3b3d3f')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.axis("off")
    if showfig:
        plt.show()
    if filename is not None:
        plt.savefig(filename)
    if return_ax:
        return fig, ax
    else:
        plt.close()

def create_circle(n_points=20, radius=5, noise_level=1e-1):
    '''
    @output:
    vertices [np,2] point coordinates 
    lines [np-1,2] per-segment point id
    '''
    angles = np.linspace(2*np.pi - 2*np.pi/n_points, 0, n_points) # need to clockwise to match the gptoolbox output vertices order 
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    segment_id = [[i%n_points,(i+1)%n_points] for i in range(n_points)]
    vertices = np.stack([x,y], axis=1)
    lines = np.stack(segment_id, axis=0)

    vertices = vertices + np.random.normal(scale=noise_level,size=vertices.shape)
    return vertices, lines

def plotMesh2D(v_in=None, l_in=None, vn_in=None, ln_in=None, rv_in=None,
               v_tgt=None, l_tgt=None, vn_tgt=None, ln_tgt=None, rv_tgt=None, sdf_error = None,   
               nr=None, gradient=None, sdf=None, showfig=False, savefig=False, figname="image.png"):
    '''
    See https://towardsdatascience.com/the-many-ways-to-call-axes-in-matplotlib-2667a7b06e06#:~:text=Rarely%2C%20as%20for%20figure%20with,can%20find%20an%20example%20here) 
    to understand more.
    @input
    - data: list of list of [vertices, lines] data in which 
        - vertices: numpy array of shape [nv, 3] of ng groups of vertices to be visualized with different color 
        - lines: [ng, nl, 2]
    '''
    #>>> open a figure
    n_rows = 1
    n_cols = 0
    if v_in is not None: n_cols+=1 
    if v_tgt is not None: n_cols+=1 
    fig = plt.figure()
    fig.set_size_inches(20, 10.5)
    ax = fig.add_subplot(n_rows, n_cols, 1)
    canvas = FigureCanvas(fig)

    #>>> plot input mesh 
    #> get axes
    ax.set_aspect('equal', adjustable='box')
    
    #> get data
    v = v_in
    l = l_in
    
    #> set axes range
    rg = v.max() - v.min()
    ax.set_xlim(v.min() - rg/4, v.max() + rg/4)
    ax.set_ylim(v.min() - rg/4, v.max() + rg/4)
    vtx = v[l,:]
    x = vtx[:,:,0].reshape((-1,1))
    y = vtx[:,:,1].reshape((-1,1))
    ax.plot(x, y, linewidth=1, zorder=0)

    line_centers = np.mean(v[l,:],axis=1)
    if sdf_error is not None:
        for i in range(line_centers.shape[0]):
            ax.annotate("{:.2f}".format(sdf_error[i]), line_centers[i])
            ax.annotate(i, line_centers[i]-np.array([0.3,0]),color='r')

    #> visualize normals
    if vn_in is not None:
        ax.quiver(v[:,0],v[:,1],vn_in[:,0],vn_in[:,1])
    if ln_in is not None:
        line_centers = np.mean(v[l,:],axis=1)
        ax.quiver(line_centers[:,0],line_centers[:,1],ln_in[:,0],ln_in[:,1])

    #>>> plot rays on input mesh
    if rv_in is not None:
        # print(rv_in.shape)
        rv_in = rv_in.reshape(-1,2) #[ray0p0,ray0p1,ray1p0,ray1p1,...]
        # print(rv_in.shape)
        rl_in = np.array([[i*2,i*2+1] for i in range(rv_in.shape[0]//2)])
        for i in range(rl_in.shape[0]//nr):
            v = rv_in
            l = rl_in[i*nr:(i+1)*nr,]
            vtx = v[l,:]
            x = vtx[:,:,0].reshape((-1,1))
            y = vtx[:,:,1].reshape((-1,1))
            ax.plot(x,y,linewidth=0.5,color='orange',zorder=1)

    #> visualize gradients
    if gradient is not None:
        gradient = - gradient
        ax.quiver(v_in[:,0], v_in[:,1], gradient[:,0], gradient[:,1], 
                  angles='xy', 
                  scale_units='xy', 
                  scale=0.5,zorder=2)

    #>>> plot target mesh
    if v_tgt is not None and l_tgt is not None:
        #> get axes
        ax = fig.add_subplot(n_rows, n_cols, 2)
        ax.set_aspect('equal', adjustable='box')
    
        #> get data
        v = v_tgt
        l = l_tgt
    
        #> set axes range
        rg = v.max() - v.min()
        ax.set_xlim(v.min() - rg/4, v.max() + rg/4)
        ax.set_ylim(v.min() - rg/4, v.max() + rg/4)
        vtx = v[l,:]
        x = vtx[:,:,0].reshape((-1,1))
        y = vtx[:,:,1].reshape((-1,1))
        ax.plot(x,y,linewidth=1)

        #>>> plot rays on target mesh 
        if rv_tgt is not None:
            #> get data
            rv_tgt = rv_tgt.reshape(-1,2) #[ray0p0,ray0p1,ray1p0,ray1p1,...]
            rl_tgt = np.array([[i*2,i*2+1] for i in range(rv_tgt.shape[0]//2)]) 
            for i in range(rv_tgt.shape[0]//nr):
                v = rv_tgt
                l = rl_tgt[i*nr:(i+1)*nr,]
                vtx = v[l,:]
                x = vtx[:,:,0].reshape((-1,1))
                y = vtx[:,:,1].reshape((-1,1))
                ax.plot(x,y,linewidth=0.5,color='orange')

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname)
    
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    image = np.transpose(image, (2,0,1))
    plt.close()
    return image