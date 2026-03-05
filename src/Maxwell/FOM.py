import h5py
import sys
import scipy
import math
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
import pyvista as pv # also need imageio


# Need:
# pip install numpy,scipy,h5py,pyvista,imageio,tqdm

def evaluate_at_cell_center(hdf5_group,
                            coeffs):

  gids = np.array(hdf5_group['gids'])
  basis = np.array(hdf5_group['basis'])
  return np.einsum('ciqd,ci->cqd',basis,coeffs[gids])

def read_mesh(hdf5_group):
  cells = np.array(hdf5_group['cells'])
  points = np.array(hdf5_group['points'])
  cell_types = np.array(hdf5_group.attrs['count']*[pv.CellType.TETRA])
  return pv.UnstructuredGrid(cells,
                             cell_types,
                             points)

def read_hdf5_sparse(hdf5_group):
  data = hdf5_group['data']
  indptr = hdf5_group['indptr']
  indices = hdf5_group['indices']
  shape = hdf5_group.attrs[f'shape']
  return scipy.sparse.csr_matrix((data,indices,indptr),shape)

def assemble_current(source_func,hdf5_group):
  quad_coords = hdf5_group['coords'] # type: ignore
  weighted_basis  = hdf5_group['weighted_basis'] # type: ignore
  gids  = hdf5_group['gids'] # type: ignore
  func_values = source_func(quad_coords)
  elmt_vec = np.einsum('ciqd,cqd->ci',weighted_basis,func_values)
  vec = np.zeros(np.amax(gids)+1)  # allocate
  np.add.at(vec,gids,elmt_vec)     # assemble
  return vec

def smooth_pulse(t, T):
    if t < 0.0 or t > T:
        return 0.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / T))

def decaying_gaussian_current(t, x, xmid, width, dir, T):
    scaling = smooth_pulse(t,T)
    #scaling = np.sin(math.pi*(1.0-t)**4) if t<1 else 0.0
    dist2 = np.sum((x-xmid)**2,axis=-1)
    result = scaling * np.exp(-dist2/(2.0 * width))

    return result[...,None] * dir

def plot_field(mesh_full,name,pl=None):
  dargs = dict(
      scalars=name,
      cmap="jet",
      show_scalar_bar=True,
      lighting=False,
  )
  
  plane_origin = [0.5, 0.5, 0.5]  # Origin of the plane
  plane_normal = [0, 0, 1]  # Normal vector of the plane (z-axis)
  mesh = mesh_full.slice(normal=plane_normal, origin=plane_origin)

  if pl==None:
    pl = pv.Plotter(shape=(2, 2))
  pl.subplot(0, 0)
  pl.add_mesh(mesh, **dargs)
  pl.add_text(f"{name} Magnitude", color='k')
  pl.subplot(0, 1)
  pl.add_mesh(mesh.copy(), component=0, **dargs)
  pl.add_text(f"{name}X", color='k')
  pl.add_scalar_bar()
  pl.subplot(1, 0)
  pl.add_mesh(mesh.copy(), component=1, **dargs)
  pl.add_text(f"{name}Y", color='k')
  pl.add_scalar_bar()
  pl.subplot(1, 1)
  pl.add_mesh(mesh.copy(), component=2, **dargs)
  pl.add_text(f"{name}Z", color='k')
  pl.add_scalar_bar()

  pl.link_views()
  pl.camera_position = 'iso'
  pl.background_color = 'white'

  return pl

def build_gif_videos_from_list(mesh,file_hdf5,prefix,timesteps,states):

  E_coeffs = states[0][0]
  B_coeffs = states[0][1]
  mesh.cell_data['E'] = evaluate_at_cell_center(file_hdf5['Eeval'],E_coeffs)
  mesh.cell_data['B'] = evaluate_at_cell_center(file_hdf5['Beval'],B_coeffs)

  plt_E = plot_field(mesh,'E')
  plt_B = plot_field(mesh,'B')

  plt_E.open_gif(f'{prefix}-E.gif',fps=5)
  plt_B.open_gif(f'{prefix}-B.gif',fps=5)

  plt_E.write_frame()
  plt_B.write_frame()

  # you can visualize it
  #plt_E.show()
  #plt_B.show()

  # you can build a GIF
  for t,state in tqdm(zip(timesteps[1:],states[1:]),desc='  ==> making movie (timesteps)'):
    E_coeffs = state[0]
    B_coeffs = state[1]

    mesh.cell_data['E'] = evaluate_at_cell_center(file_hdf5['Eeval'],E_coeffs)
    mesh.cell_data['B'] = evaluate_at_cell_center(file_hdf5['Beval'],B_coeffs)

    plt_E.clear()
    plt_E = plot_field(mesh,'E',plt_E)
    plt_E.write_frame()

    plt_B.clear()
    plt_B = plot_field(mesh,'B',plt_B)
    plt_B.write_frame()

  plt_E.close()
  plt_B.close()

class MaxwellSim:
  def __init__(self,file_hdf5):
    self.file_hdf5 = file_hdf5
    self.mesh = read_mesh(file_hdf5['Mesh'])
    self.emass = read_hdf5_sparse(file_hdf5['Emass'])
    self.bmass = read_hdf5_sparse(file_hdf5['Bmass'])
    self.stcurl = read_hdf5_sparse(file_hdf5['StCurl'])
    self.wkcurl = read_hdf5_sparse(file_hdf5['WkCurl'])
    self.emass_lu = sparse.linalg.splu(self.emass.tocsc())

    self.ebc_gids = np.nonzero(self.emass.diagonal()==1.)

    # mid = np.array([1.0,1.0,1.0])
    # width = 1e-2
    # dir = np.array([1.,0.,0.])

    # source params set later
    self.mid = None
    self.width = None
    self.dir = None

  def set_source(self, mid, width, dir):
        self.mid = np.asarray(mid, dtype=float)
        self.width = float(width)
        self.dir = np.asarray(dir, dtype=float)

  def jfunc(self, t, x, T):
      if self.mid is None or self.width is None or self.dir is None:
        raise RuntimeError("Source not set. Call sim.set_source(mid,width,dir) first.")
      return decaying_gaussian_current(t, x, self.mid, self.width, self.dir, T)
  
  def timeLoop(self,t0,tf,nsteps,record_freq=1,E0=None,B0=None):
    if E0==None: E0 = np.zeros(self.emass.shape[0])
    if B0==None: B0 = np.zeros(self.bmass.shape[0])

    dt = (tf-t0)/nsteps

    En = E0
    Bn = B0
    #jn = self.emass @ En + 0.5 * dt * self.wkcurl @ Bn
    
    time_history = []
    state_history = []
    for i in tqdm(range(nsteps),desc='  ==> Maxwell sim (timesteps)'):
      t = t0 + i * dt
      if i % record_freq==0:
        state_history.append((En,Bn))
        time_history.append(t)
      
      En,Bn = self.takeStep(En,Bn,t,dt,tf)

    if nsteps % record_freq==0:
      state_history.append((En,Bn))
      time_history.append(t0+nsteps*dt)

    return state_history, time_history

  def takeStep(self, E_n, B_n, t0 : float, dt : float, tf: float):

    # This is based on a Velocity Verlet time integrator: 
    # both E & B are collocated at time nodes
    #######################################################

    emass = self.emass
    stcurl = self.stcurl
    wkcurl = self.wkcurl
    emass_lu = self.emass_lu

    # build weak form (RHS)
    ampere_rhs = emass @ E_n + 0.5 * dt * wkcurl @ B_n

    
    # evaluate current
    source_j = lambda x: self.jfunc(t0, x, tf) # set up lambda
    current_source = assemble_current(source_j,self.file_hdf5['Current_Construction'])
    ampere_rhs -= 0.5 * dt * current_source

    # evaluate dirichlet
    ampere_rhs[self.ebc_gids] = 0.0 # set BCs

    # solve ampere
    E_half = emass_lu.solve(ampere_rhs)

    # faraday solve
    #########################

    # # solve faraday
    B_full = B_n - dt * stcurl @ E_half # this is strong form only

    # ampere solve
    #########################

    # build weak form (RHS)
    ampere_rhs = emass @ E_half + 0.5 * dt * wkcurl @ B_full

    # evaluate current
    source_j = lambda x: self.jfunc(t0+1.*dt,x,tf) # set up lambda
    current_source = assemble_current(source_j,self.file_hdf5['Current_Construction'])
    ampere_rhs -= 0.5 * dt * current_source

    # evaluate dirichlet
    ampere_rhs[self.ebc_gids] = 0.0 # set BCs

    # solve ampere
    E_full = emass_lu.solve(ampere_rhs)

    return E_full, B_full

# filename = 'src/Maxwell/maxwell_data.hdf5'
# print('Usage: <cmd> <HDF5 filename>')
# print('  note that the filename is optional and will default to "maxwell_data.hdf5"')
# print()
# if len(sys.argv)==2:
#   filename = sys.argv[1]

# print('... Loading HDF5 File')
# file_hdf5 = h5py.File(filename, 'r')

# print('... Building Simulation Structure')
# sim = MaxwellSim(file_hdf5)

# print('... Time Stepping')
# states,timesteps = sim.timeLoop(t0=0.0,tf=2.5,nsteps=120,record_freq=2)

# print('... Movie Magic')
# build_gif_videos_from_list(sim.mesh,file_hdf5,'sim',timesteps,states)