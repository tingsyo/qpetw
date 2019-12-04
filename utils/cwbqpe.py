class cwbqpe:
    '''Class for processing CWB pre-QC QPE data.'''
    def __init__(self, file=None, data=None):
        import os, gzip, struct
        import numpy as np
        self.uri = file
        self.header = None
        self.data = data
        
    def help(self):
        print("This toolset provides functions accessing CWB QPESUMS data. \nThe data is 494972 bytes binary stored in gzip format. The first 170 bytes is the header, and the latter part is the QPE results on a (441x561) surface.\n")
    
    def load_data(self, file=None):
        import os, gzip, struct
        import numpy as np
        # Check data file
        if (self.uri is None):
            if (file is None) or (not os.path.isfile(file)):
                print('[Error] The data file is not specified or does not exist.')
                return(None)
            else:
                self.uri = file
        # Load data
        with gzip.open(self.uri, 'rb') as f:
            raw = f.read()
        # Parse header
        self.header = self.parse_header(raw[:170])
        self.data = np.array(struct.unpack('247401h', raw[170:])).reshape(self.header['ny'], self.header['nx'])
        # Scale data
        self.data = self.data/self.header['var_scale']
        return(0)
    
    def parse_header(self, raw):
        import os, gzip, struct
        header = {}
        # Time information
        header['year'] = struct.unpack('i', raw[:4])[0]
        header['month'] = struct.unpack('i', raw[4:8])[0]
        header['day'] = struct.unpack('i', raw[8:12])[0]
        header['hour'] = struct.unpack('i', raw[12:16])[0]
        header['minute'] = struct.unpack('i', raw[16:20])[0]
        header['second'] = struct.unpack('i', raw[20:24])[0]
        # Data dimension
        header['nx'] = struct.unpack('i', raw[24:28])[0]
        header['ny'] = struct.unpack('i', raw[28:32])[0]
        header['nz'] = struct.unpack('i', raw[32:36])[0]
        # Projection and lat/lon
        header['proj'] = struct.unpack('4s', raw[36:40])[0].decode('ISO-8859-1')
        header['map_scale'] = struct.unpack('i', raw[40:44])[0]
        header['projlat1'] = struct.unpack('i', raw[44:48])[0]
        header['projlat2'] = struct.unpack('i', raw[48:52])[0]
        header['projlon'] = struct.unpack('i', raw[52:56])[0]
        header['alon'] = struct.unpack('i', raw[56:60])[0]
        header['alat'] = struct.unpack('i', raw[60:64])[0]
        # Delta in x-y-z
        header['pxy_scale'] = struct.unpack('i', raw[64:68])[0]
        header['dx'] = struct.unpack('i', raw[68:72])[0]
        header['dy'] = struct.unpack('i', raw[72:76])[0]
        header['dxy_scale'] = struct.unpack('i', raw[76:80])[0]
        header['zht'] = struct.unpack('i', raw[80:84])[0]
        header['z_scale'] = struct.unpack('i', raw[84:88])[0]
        header['i_bb_mode'] = struct.unpack('i', raw[88:92])[0]
        # Quality information
        unkn01,unkn02,unkn03,unkn04,unkn05,unkn06,unkn07,unkn08,unkn09 = struct.unpack('iiiiiiiii', raw[92:128])
        # Variable information
        header['varname'] = struct.unpack('20s', raw[128:148])[0].decode('ISO-8859-1')
        header['varunit'] = struct.unpack('6s', raw[148:154])[0].decode('ISO-8859-1')
        header['var_scale'] = struct.unpack('i', raw[154:158])[0]
        header['missing'] = struct.unpack('i', raw[158:162])[0]
        header['nradar'] = struct.unpack('i', raw[162:166])[0]
        header['mosradar'] = struct.unpack('4s', raw[166:170])[0].decode('ISO-8859-1')
        #
        return(header)

    def find_nearest_value(self, lon, lat):
        import os, gzip, struct
        import numpy as np
        ''' Find the closest point in the dataset to the specified lon/lat.'''
        # Check data file
        if (self.header is None):
            print('[Error] The object has not yet been initialized.')
            return(None)
        # Derive the coordinate of the data object
        lon0 = self.header['alon']/self.header['map_scale']
        lat1 = self.header['alat']/self.header['map_scale']
        dx = self.header['dx']/self.header['dxy_scale']
        dy = self.header['dy']/self.header['dxy_scale']
        lon1 = lon0 + (self.header['nx']-1)*dx
        lat0 = lat1 - (self.header['ny']-1)*dy
        lons = np.linspace(lon0, lon1, self.header['nx'])
        lats = np.linspace(lat0, lat1, self.header['ny'])
        # Check boundaries
        if (lon<lon0) or (lon>lon1) or (lat<lat0) or (lat>lat1):
            print("Specified lon/lat is outside of the data boundary: "+
                  str(lon0)+"~"+str(lon1)+", "+str(lat0)+"~"+str(lat1))
            return(None)
        # Find neighbors
        ilonr = np.where(lons>lon)[0][0]
        ilonl = np.where(lons<=lon)[0][-1]
        ilatu = np.where(lats>lat)[0][0]
        ilatd = np.where(lats<=lat)[0][-1]
        # Determin the closest point
        if (lon - lons[ilonl]) <= (lons[ilonr] - lon):
            ilon = ilonl
        else:
            ilon = ilonr
        if (lat - lats[ilatd]) <= (lats[ilatu] - lat):
            ilat = ilatd
        else:
            ilat = ilatu
        #
        return((lons[ilon], lats[ilat], self.data[ilat,ilon]))

    def find_interpolated_value(self, lon, lat):
        import os, gzip, struct
        import numpy as np
        ''' Find the closest points and interpolate to the specified lon/lat.'''
        # Check data file
        if (self.header is None):
            print('[Error] The object has not yet been initialized.')
            return(None)
        # Derive the coordinate of the data object
        lon0 = self.header['alon']/self.header['map_scale']
        lat1 = self.header['alat']/self.header['map_scale']
        dx = self.header['dx']/self.header['dxy_scale']
        dy = self.header['dy']/self.header['dxy_scale']
        lon1 = lon0 + (self.header['nx']-1)*dx
        lat0 = lat1 - (self.header['ny']-1)*dy
        lons = np.linspace(lon0, lon1, self.header['nx'])
        lats = np.linspace(lat0, lat1, self.header['ny'])
        # Check boundaries
        if (lon<lon0) or (lon>lon1) or (lat<lat0) or (lat>lat1):
            print("Specified lon/lat is outside of the data boundary: "+
                  str(lon0)+"~"+str(lon1)+", "+str(lat0)+"~"+str(lat1))
            return(None)
        # Find neighbors
        ilonr = np.where(lons>lon)[0][0]
        ilonl = np.where(lons<=lon)[0][-1]
        ilatu = np.where(lats>lat)[0][0]
        ilatd = np.where(lats<=lat)[0][-1]
        # Interpolate
        def bilinear_interpolation(x, y, x1, x2, y1, y2, z):
            '''Bilinear interpolation, ref:https://en.wikipedia.org/wiki/Bilinear_interpolation'''
            A = np.array([[1,x1,y1,x1*y1],[1,x1,y2,x1*y2],[1,x2,y1,x2*y1],[1,x2,y2,x2*y2]])
            a = np.linalg.solve(A,z)
            fxy = a[0] + a[1]*x + a[2]*y + a[3]*x*y
            return(fxy)
        #
        neighbours = [self.data[ilatd,ilonl], self.data[ilatu,ilonl], self.data[ilatd,ilonr], self.data[ilatu,ilonr]]
        value = bilinear_interpolation(lon, lat, lons[ilonl], lons[ilonr], lats[ilatd], lats[ilatu], neighbours)
        return(value)
