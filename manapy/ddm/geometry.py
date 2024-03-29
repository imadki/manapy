
class Cell():
    """ """
    __slots__=['_nbcells', '_nodeid', '_faceid', '_cellfid', '_cellnid', '_halonid', '_ghostnid', '_haloghostnid', 
               '_haloghostcenter', '_center', '_volume', '_nf', '_globtoloc', '_loctoglob', '_tc', '_periodicnid', '_periodicfid',
               '_shift', '_orthocenter']
    def __init__(self):
        pass
        
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nodeid(self):
        return self._nodeid
    
    @property
    def faceid(self):
        return self._faceid
    
    @property
    def cellfid(self):
        return self._cellfid
    
    @property
    def cellnid(self):
        return self._cellnid
    
    @property
    def halonid(self):
        return self._halonid
    
    @property
    def ghostnid(self):
        return self._ghostnid
    
    @property
    def haloghostnid(self):
        return self._haloghostnid
    
    @property
    def haloghostcenter(self):
        return self._haloghostcenter
    
    @property
    def center(self):
        return self._center
    
    @property
    def volume(self):
        return self._volume
    
    @property
    def nf(self):
        return self._nf
    
    @property
    def globtoloc(self):
        return self._globtoloc
    
    @property
    def loctoglob(self):
        return self._loctoglob
    
    @property
    def tc(self):
        return self._tc
    
    @property
    def periodicnid(self):
        return self._periodicnid
    
    @property
    def periodicfid(self):
        return self._periodicfid
    
    @property
    def shift(self):
        return self._shift
            
class Node():
    """ """
    __slots__= ['_nbnodes', '_vertex', '_name', '_oldname', '_cellid', '_ghostid', '_haloghostid', '_ghostcenter', '_haloghostcenter', '_ghostfaceinfo', 
                '_haloghostfaceinfo', '_loctoglob', '_halonid', '_nparts', '_periodicid', '_R_x', '_R_y', '_R_z', '_number', 
                '_lambda_x', '_lambda_y', '_lambda_z']
     
    def __init__(self, nbnodes=None):
        pass
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def vertex(self):
        return self._vertex
    
    @property
    def name(self):
        return self._name
    
    @property
    def oldname(self):
        return self._oldname
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def ghostid(self):
        return self._ghostid
    
    @property
    def haloghostid(self):
        return self._haloghostid
    
    @property
    def ghostcenter(self):
        return self._ghostcenter
    
    @property
    def haloghostcenter(self):
        return self._haloghostcenter
    
    @property
    def ghostfaceinfo(self):
        return self._ghostfaceinfo
    
    @property
    def haloghostfaceinfo(self):
        return self._haloghostfaceinfo
    
    @property
    def loctoglob(self):
        return self._loctoglob
    
    @property
    def halonid(self):
        return self._halonid
    
    @property
    def nparts(self):
        return self._nparts
    
    @property
    def periodicid(self):
        return self._periodicid
    
    @property
    def R_x(self):
        return self._R_x
    
    @property
    def R_y(self):
        return self._R_y
    
    @property
    def R_z(self):
        return self._R_z
    
    @property
    def number(self):
        return self._number
    
    @property
    def lambda_x(self):
        return self._lambda_x
    
    @property
    def lambda_y(self):
        return self._lambda_y
    
    @property
    def lambda_z(self):
        return self._lambda_z

class Face():
    """ """
    __slots__= ['_nbfaces', '_nodeid', '_cellid', '_name', '_oldname', '_normal', '_mesure', '_center', '_ghostcenter', '_oppnodeid', '_halofid',
                '_halofid', '_param1', '_param2', '_param3', '_param4', '_f_1', '_f_2', '_f_3', '_f_4', '_airDiamond', 
                '_dist_ortho', '_tangent', '_binormal']
                
    def __init__(self):
        pass
        
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nodeid(self):
        return self._nodeid
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def name(self):
        return self._name
    
    @property
    def oldname(self):
        return self._oldname
    
    @property
    def normal(self):
        return self._normal
    
    @property
    def mesure(self):
        return self._mesure
    
    @property
    def center(self):
        return self._center
    
    @property
    def dist_ortho(self):
        return self._dist_ortho
    
    
    @property
    def ghostcenter(self):
        return self._ghostcenter
    
    
    @property
    def oppnodeid(self):
        return self._oppnodeid
    
    @property
    def halofid(self):
        return self._halofid
    
    @property
    def param1(self):
        return self._param1
    
    @property
    def param2(self):
        return self._param2
    
    @property
    def param3(self):
        return self._param3
    
    @property
    def param4(self):
        return self._param4
    
    @property
    def f_1(self):
        return self._f_1
    
    @property
    def f_2(self):
        return self._f_2
    
    @property
    def f_3(self):
        return self._f_3
    
    @property
    def f_4(self):
        return self._f_4
    
    @property
    def airDiamond(self):
        return self._airDiamond
    @property
    def tangent(self):
        return self._tangent
    @property
    def binormal(self):
        return self._binormal
    
    
    # @property
    # def K(self):
    #     return self._K
        
class Halo():
    """ """
    __slots__= ['_halosint', '_halosext', '_neigh', '_centvol', '_faces', '_nodes', '_sizehaloghost', '_scount', '_rcount', '_sdepl', 
                '_rdepl','_indsend', '_comm_ptr', '_requests']
    
    def __init__(self):
        pass
        
    @property
    def halosint(self):
        return self._halosint
    
    @property
    def halosext(self):
        return self._halosext
    
    @property
    def neigh(self):
        return self._neigh
    
    @property
    def centvol(self):
        return self._centvol
    
    @property
    def faces(self):
        return self._faces
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def sizehaloghost(self):
        return self._sizehaloghost
    
    @property
    def scount(self):
        return self._scount
    
    @property
    def rcount(self):
        return self._rcount
    
    @property
    def indsend(self):
        return self._indsend
    
    
    @property
    def comm_ptr(self):
        return self._comm_ptr
    
    @property
    def requests(self):
        return self._requests