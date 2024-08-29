def read_world(filename):
    # Reads the world definition and returns a list of landmarks, our 'map'.
    # 
    # The returned dict contains a list of landmarks each with the
    # following information: {id, [x, y]}

    landmarks = dict()

    f = open(filename)
    
    for line in f:

        line_s  = line.split('\n')    
        line_spl  = line_s[0].split(' ')
        landmarks[int(line_spl[0])] = [float(line_spl[1]),float(line_spl[2])]      

    return landmarks

def read_pos(filename):
    # Reads the intial position of robot from pos.dat
    # The returned following information: [x, y, theta]
    f = open(filename)
    
    for line in f:
        line_s  = line.split('\n')    
        line_spl  = line_s[0].split(' ')
        pos = [float(line_spl[0]),float(line_spl[1]),float(line_spl[2])]      

    return pos

actual_robot_pos = read_pos("./pos.dat")
