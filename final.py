
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.sparse import linalg, eye
import scipy.linalg as LA


with open('approval_polllist.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    approve = []
    disapprove = []
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
            continue
        else:
            approve.append(row[11])
            disapprove.append(row[12])


dataset = pd.read_csv('approval_polllist.csv')
dataset.plot(x='approve', y='disapprove', style='.', legend=None)
plt.title('approval vs disapproval ratings for trump')
plt.xlabel('approve')
plt.ylabel('disapprove')
plt.show()


X = dataset['approve'].values.reshape(-1, 1)
y = dataset['disapprove'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
reg = regressor.fit(X_train, y_train)
y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test.
                   flatten(), 'Predicted': y_pred.flatten()})
print(df)
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title(
    'estimated linear relationship between approval and dissaproval rating of trump')
plt.xlabel('approve')
plt.ylabel('disapprove')
plt.show()
df1 = df.head(75)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('How accuracte is linear regression in determining trumps dissaproval rating given approval rating (first 100 shown here)')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


def data_maker(X, y):
    X_list = []
    y_list = []
    for i in X:
        X_list.append(i[0])
    for i in y:
        y_list.append(i[0])

    twoD_X = np.array([[0, 0]])
    for i in range(len(X_list)):
        twoD_X = np.append(twoD_X, [[X_list[i], y_list[i]]], axis=0)
    twoD_X = np.delete(twoD_X, 0, 0)
    return X, y, twoD_X


def data_plotter(X_list, y_list, twoD_X):

    pca = PCA(n_components=2)
    twoD_X = pca.fit(twoD_X)
    ax = 0

    plt.scatter(X_list, y_list, alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * np.sqrt(length*4)
        ax = draw_vector(pca.mean_, pca.mean_ + v)
    # plt.axes('equal')
    #confidence_ellipse(x, y, ax, edgecolor = 'r', n_std = 1)

    plt.show()


def draw_vector(v0, v1, ax=None):
    ax = plt.gca()

    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    return ax


X_list, y_list, twoD_X = data_maker(X, y)
data_plotter(X_list, y_list, twoD_X)


def euclidean_dist(data):
    # euc_dist will eventually store the distance from each point/row to each
    # other point/row
    euc_dist = np.zeros(shape=(data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            #dist = sqrt((x_1 - x_2_)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2)
            subtraction = data[i, :] - data[j, :]
            squared_subtraction = subtraction ** 2
            squared_subtraction_sum = sum(squared_subtraction)
            squared_subtraction_sum_sqrt = np.sqrt(squared_subtraction_sum)
            euc_dist[i, j] = squared_subtraction_sum_sqrt
    return euc_dist + euc_dist.T


def multidimentional_scaling(output_dim, euc_dist):
    # multidimensional scaling by taking two parameters as inputs (the N x N
    # matrix of distances, and 'dims' which is the desired number of output
    # dimensions), and output a matrix of size N x dims, where each row is
    # a reconstructed data point.

    # NOTE: I know my MDS function is a lot longer than 5 lines :(
    # I really struggled with this and spacing out different computations
    # really helped me stop having bugs/issues/errors.  I apologize for that.

    # number of rows in our data (number of data points)
    data_size = euc_dist.shape[1]
    # construct identity equal to # of rows in data
    matrix_I = np.eye(data_size)
    # Constructs a hadamar of our euclidean distance matrix
    euc_dist_H = euc_dist * euc_dist
    # Prep for --> singular value decompostion/eigendecomposition
    matrix_A = matrix_I - np.ones([data_size, data_size])/data_size
    matrix_left = np.matmul(matrix_A, euc_dist_H)
    matrix_SVD = np.matmul(matrix_left, matrix_A)
    # perform SVD, obtain eigenvalues/singular values
    U, sigma, V = np.linalg.svd(matrix_SVD, full_matrices=True)
    # extra check
    if np.allclose(matrix_SVD, np.dot(U * sigma, V)) != True:
        print("Error!!!")
    else:
        print("svd working properly")
    # we are always going to reduce dimentionality to 2D,
    # Thus our final matrix_MDS will be concat(X, Y)
    X = np.sqrt(np.diag(sigma[0:output_dim]))
    Y = V[:output_dim, :]
    matrix_MDS = np.matmul(X, Y).T
    return matrix_MDS


def knn(euc_dist, k, title):
    print(f'\nStarting computations for knn with: {title}')
    neighbors = np.zeros(euc_dist.shape)
    euc_dist_copy = np.copy(euc_dist)
    for index, row in enumerate(euc_dist_copy):
        count = 0
        while count < k:

            min_index = np.argmin(row)
            if row[min_index] == 0:
                row[min_index] = np.inf
            elif count < k and row[min_index] != 0:
                neighbors[index, min_index] = 1
                row[min_index] = np.inf
                count += 1
    return neighbors


def epsilon_ball(euc_dist, k, title):
    print(f'\nStarting computations for epsilon_ball with: {title}')
    neighbors = np.zeros(euc_dist.shape)
    for row_index, row in enumerate(euc_dist):
        for col_index, col in enumerate(row):
            if k - col >= 0 and col != 0:
                neighbors[row_index, col_index] = 1
    return neighbors


def isomap(euc_dist, neighbors, output_dim, title):
    """
    outputs a complete NxN matrix of intrinsic distances by implementing the procedure 
    in step 2. 
    """
    print(f'\nStarting computations for isomap with: {title}')
    euc_dist_copy = np.copy(euc_dist)
    euc_dist_copy = euc_dist_copy * neighbors
    euc_dist_copy[euc_dist_copy == 0] = np.inf
    for k in range(euc_dist_copy.shape[0]):
        for i in range(euc_dist_copy.shape[0]):
            for j in range(euc_dist_copy.shape[0]):
                #print(min(euc_dist_copy[i][k] + euc_dist_copy[i][k], euc_dist_copy[k][j]))
                euc_dist[i, j] = min(
                    euc_dist_copy[i][k] + euc_dist_copy[i][k], euc_dist_copy[k][j])
    euc_dist_copy[euc_dist_copy == np.inf] = 0
    D_iso = multidimentional_scaling(output_dim, euc_dist_copy)
    return D_iso


x = np.copy(twoD_X)
euc_dist = euclidean_dist(x)
MDS_MATRIX = multidimentional_scaling(2, euc_dist)
plt.plot(MDS_MATRIX[:, 0], MDS_MATRIX[:, 1], 'o', color='green', markersize=5)
plt.style.use('seaborn-whitegrid')
plt.show()
neighbors_knn = knn(euc_dist, 5, "Trump Approval Rating")
neighbors_ball = epsilon_ball(euc_dist, 5, "Trump Approval Rating")
final = isomap(euc_dist, neighbors_ball, 2, "Trump Approval Rating")
plt.plot(final[:, 0], final[:, 1], 'o', color='red', markersize=5)
plt.style.use('seaborn-whitegrid')
plt.show()


def LLE(euc_dist, neighbors_knn):
    mat_A = np.eye(*neighbors_knn.shape) - neighbors_knn
    mat_A = (mat_A.T).dot(mat_A)
    e_vals, e_vecs = LA.eig(mat_A)
    index = np.argsort(e_vals)
    return e_vecs[:, index], np.sum(e_vals)


final_DAT, co1 = LLE(euc_dist, neighbors_knn)
plt.plot(final_DAT[:, 0], final_DAT[:, 1], 'o', color='green', markersize=5)
plt.style.use('seaborn-whitegrid')
plt.title("Analysing Trump's approval rating using LLE embedding")
plt.show()


def Lap_eig(w):
    return LA.eig(np.diag(np.sum(w, axis=1))-w)


idx = np.random.randint(len(X), size=30)
data = np.c_[X, y]
data = data[idx, :]
euc_dist_dat = euclidean_dist(data)
w_mat = knn(euc_dist_dat, 7, '')
lap_eig = Lap_eig(w_mat)
plt.scatter(lap_eig[1][0], lap_eig[1][1],  color='gray')


def euclidean_dist(data):
    euc_dist = np.zeros(shape=(data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            #dist = sqrt((x_1 - x_2_)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2)
            subtraction = data[i, :] - data[j, :]
            squared_subtraction = subtraction ** 2
            squared_subtraction_sum = sum(squared_subtraction)
            squared_subtraction_sum_sqrt = np.sqrt(squared_subtraction_sum)
            euc_dist[i, j] = squared_subtraction_sum_sqrt
    return euc_dist + euc_dist.T


def knn(euc_dist, k, title):
    print(f'\nStarting computations for knn with: {title}')
    neighbors = np.zeros(euc_dist.shape)
    euc_dist_copy = np.copy(euc_dist)
    for index, row in enumerate(euc_dist_copy):
        count = 0
        while count < k:

            min_index = np.argmin(row)
            if row[min_index] == 0:
                row[min_index] = np.inf
            elif count < k and row[min_index] != 0:
                neighbors[index, min_index] = 1
                row[min_index] = np.inf
                count += 1
    return neighbors


def epsilon_ball(euc_dist, k, title):
    print(f'\nStarting computations for epsilon_ball with: {title}')
    neighbors = np.zeros(euc_dist.shape)
    for row_index, row in enumerate(euc_dist):
        for col_index, col in enumerate(row):
            if k - col >= 0 and col != 0:
                neighbors[row_index, col_index] = 1
    return neighbors


def find_lower(edges, index):
    return np.intersect1d(np.array([i for i in range(index)]), np.where(edges[index, :] > 0)[0])


def low(boundary, ind):
    N = boundary.shape[0]
    i = N-1
    while i >= 0 and boundary[i, ind] == 0:
        i -= 1
    return i


def draw_simplexes(Data, VR, title, save_root, count):
    """Data is the original data, VR is the Vietoris-Rips Complex, title is the intended title 
    for each image drawn, save_root is the directory for images to be saved, and count is the 
    the sequence order (function usually ran in a for loop).
    This function is not equipped to draw anything larger than a 2-simplex."""

    plt.figure()
    plt.scatter(Data[:, 0], Data[:, 1])

    inds_1 = VR[1]
    for i in range(inds_1.shape[1]):
        plt.plot(Data[inds_1[:, i], 0], Data[inds_1[:, i], 1], color='k')

    inds_2 = VR[2]
    for j in range(inds_2.shape[1]):
        plt.fill(Data[inds_2[:, j], 0],
                 Data[inds_2[:, j], 1], alpha=0.5, color='r')

    if count < 10:
        num = "0" + str(count)
    else:
        num = str(count)
    title = title + f'. Num={num}'
    plt.title(title)
    plt.savefig(save_root + num + ".png", format='png')
    return


def circle(rad, center, N):
    coords = np.zeros([N, 2])
    for i in range(N):
        alph = 2*np.pi*np.random.rand()
        r = rad*((1 + np.random.rand())/10)
        coords[i, 0] = r*np.cos(alph) + center[0]
        coords[i, 1] = r*np.sin(alph) + center[1]
    return coords


def torus(r1, r2, center, N):
    coords = np.zeros([N, 3])
    for i in range(N):
        alph = 2*np.pi*np.random.rand()
        beta = 2*np.pi*np.random.rand()
        a = r2*np.sqrt(np.random.rand())
        coords[i, 0] = (r1 + a*np.cos(alph))*np.cos(beta)
        coords[i, 1] = (r1 + a*np.cos(alph))*np.sin(beta)
        coords[i, 2] = a*np.sin(alph)
    return coords


def sphere(r, center, N):
    coords = np.zeros([N, 3])
    for i in range(N):
        phi = 2*np.pi*np.random.rand()
        theta = 2*np.pi*np.random.rand()
        rad = r*((1 + np.random.rand())/10)
        coords[i, 0] = rad*np.cos(theta)*np.cos(phi)
        coords[i, 1] = rad*np.cos(theta)*np.sin(phi)
        coords[i, 2] = rad*np.sin(theta)
    return coords


def construct_VietorisRips(edges, top_dim):
    """Inputs: edge matrix and highest simplicial dimension to compute (less than
    or equal to the dimensionality of data).
    Output: VR is a list of all simplexes contained in the Vietoris-Rips complex.

    For instance VR[0] would contain the np-array of all 0-simplexes (i.e. data points)
    which is just a vector.  VR[1] is the np-array of all 1-simplexes (i.e. edges) which 
    is a 2xk matrix where each column contains the index values of the two points to draw an
    edge between.  VR[2] is the np-array of all 2-simplexes which is a 3xk matrix where each
    column contains the index values of the 3 points forming the simplex, etc."""

    L = edges.shape[0]
    VR = [np.array([i for i in range(L)])]
    skel = np.empty(shape=(2, 0), dtype=int)
    for i in range(L):
        lower = find_lower(edges, i)
        if lower.shape != 0:
            for j in lower:
                skel = np.append(skel, np.array([[i], [j]]), axis=1)
    VR += [skel]
    for i in range(2, top_dim+1):
        Prev = VR[i-1]
        Faces = np.empty(shape=(i+1, 0), dtype=int)
        for j in range(Prev.shape[1]):
            N = [find_lower(edges, ind) for ind in Prev[:, j]]
            if len(N) > 0:
                s = 0
                nums = N[0]
                while s < len(N) and len(nums) != 0:
                    nums = np.intersect1d(nums, N[s])
                    s += 1
                for p in nums:
                    vec = np.append(Prev[:, j], p).reshape((i+1, 1))
                    Faces = np.append(Faces, vec, axis=1)
        VR += [Faces]
    return VR


def simplex_ordering(VR_list):
    """Input: VR_list, a list containing a filtration of VR complexes.
    Outputs: 'order_list' which is a list presenting a total ordering on all of the simplexes
    in the top filtration VR_list[-1].  Simplexes are ordered according to two rules:
        1. Each face preceeds the simplex that it is a face to, and
        2. Any simplex appearing in VR_list[i] preceeds any simplex that first appears
           in VR_list[i+1].

    order_list should have length top_dim+1, where order_list[0] will be an np-array
    containing the order of the 0-simplexes in VR_list[-1][0].  Similarly order_list[k]
    will be an np-array containing the order of the corresponding simplexes in VR_list[-1][k].

    first_appearance will have the same shape and structure as order_list, except each
    entry first_appearance[i][j] will be equal to smallest k in which the simplex appears
    in VR_list[k]."""

    L = len(VR_list)
    order_list = []
    first_appearance = []
    top_dim = len(VR_list[-1])
    N = VR_list[top_dim][0].shape[0]
    order_list += [np.array(range(N)).reshape([N, 1])]
    first_appearance += [np.zeros([N, 1])]
    s = N
    for i in range(1, top_dim):
        top_simp = VR_list[-1][i]
        n = top_simp.shape[1]
        top_simp = np.hsplit(top_simp)
        rel_order = np.zeros([n, 1])
        appearance = np.zeros([n, 1])
        count = 0
        for j in range(L):
            if s >= sum([len(p) for p in order_list]) + n:
                break
            else:
                mat = VR_list[j][i]
                siz = mat.shape[1]
                if siz > 0:
                    mat = np.hsplit(mat, siz)
                    if j != 0:
                        prev_mat = VR_list[j-1][i]
                        prev_size = prev_mat.shape(1)
                        if prev_size > 0:
                            prev_mat = np.hsplit(prev_mat, prev_size)
                            mat = [x for x in mat if any(
                                (prev_mat[i] == x).all() for i in prev_mat)]
                        siz = len(mat)
                    for k in range(siz):
                        check = [np.array_equal(mat[k], m) for m in top_simp]
                        if True in check:
                            ind = check.index(True)
                            rel_order[ind] = int(s)
                            appearance[ind] = j
                            s += 1
                            count += 1
        order_list += [rel_order]
        first_appearance += [appearance]
    return order_list, first_appearance


def find_boundary(VR_top, order_list):
    """Inputs: VR_top which is just VR_list[-1] and order_list.
    Output: the NxN matrix 'boundary' where N = sum([len(k) for k in order_list]), or rather
    the total number of simplexes appearing in the highest VR complex.  The rows and columns
    refer to the total ordering placed on all simplexes (contained in order_list).  boundary[i,j] = 1
    if simplex-i is contained in the boundary of simplex-j, and boundary[i,j] = 0 otherwise."""
    length = [len(i) for i in order_list]
    N = sum(length)
    boundry_mat = np.zeros([N, N])
    vertic = len(order_list[0])
    l = 0
    for i in range(vertic, N):
        if i >= sum([len(order_list[k]) for k in range(l+1)]):
            ind = int(np.where(order_list[l] == i)[0])
            simp = VR_top[l][:, ind]
            for j in range(length[l-1]):
                if l == 1:
                    low_simp = VR_top[l-1][j]
                    if any((low_simp == simp[s]).all() for s in range(simp.shape[0])):
                        row = order_list[l-1][j]
                        boundry_mat[row, i] = 1
                else:
                    combos = list(combinations(list(simp), l))
                    combos = [np.asarray(f) for f in combos]
                    if any((low_simp == combos[k]).all() for k in range(len(combos))):
                        row = int(order_list[l-1][j])
                        boundry_mat[row, i] = 1

    return boundry_mat


def reduce_mat(boundary):
    """Input: boundary matrix.
    Output: NxN matrix of the reduced boundary matrix.  This is a Euclidean algorithm that 
    essentially reduces boundary to get rid of redundant components.  This will make use 
    of the 'low' function presented in the Helper Functions listed above."""

    N = boundary.shape[0]
    reduced = boundary
    for j in range(N):
        l_j = low(reduced, j)
        inds = list(np.where(reduced[l_j, :] == 1)[0])
        inds = list(set(inds).intersection(set([k for k in range(j)])))
        matches = [k for k in inds if (low(reduced, k) == l_j)]
        while len(matches) > 0:
            reduced[:, j] += reduced[:, matches[0]]
            reduced[:, j] %= 2
            l_j = low(reduced, j)
            inds = list(np.where(reduced[l_j, :] == 1)[0])
            inds = list(set(inds).intersection(set([k for k in range(j)])))
            matches = [k for k in inds if (low(reduced, k) == l_j)]
    return reduced


def draw_bar_codes(r_boundary, appearance, order_list, save_root, title):
    """Inputs: reduced boundary matrix r_boundary, the first appearance of each
    simplex in 'appearance', the total ordering 'order_list', the root directory to
    save the barcode image 'save_root', and title for the barcode image 'title.'
    Outputs: 'features' which is a list containing the barcode information as small np-arrays
    indicating where to plot the horizontal lines in the bar code.  This function should also
    produce the barcode graph and also save the image to a desired directory."""
    chains = [len(k) for k in appearance]
    T = int(max(appearance[-1])) + 1
    features = []
    c = 0
    height = 1
    H = []
    ind_list = [k for k in range(r_boundary.shape[0])]
    while len(ind_list) > 0:
        current = ind_list[0]
        if current >= sum(chains[s] for s in range(c+1)):
            features += [H]
            H = []
            c += 1
            height = 1 + 0.25*c
        lower = low(r_boundary, current)
        inds = list(np.where(r_boundary[current, :] == 1)[0])
        inds = list(set(inds).intersection(set(ind_list)))
        upper = [k for k in inds if (low(r_boundary, k) == current)]
        if len(upper) > 0:
            ind_u = int(np.where(order_list[c+1] == upper[0])[0])
            death = int(appearance[c+1][ind_u][0])
            loc = np.where(ind_list == upper[0])[0]
            del ind_list[loc[0]]
        else:
            death = T
        if lower == -1:
            ind_l = int(np.where(order_list[c] == current)[0])
            birth = int(appearance[c][ind_l][0])
        del ind_list[0]
        if death > birth:
            l = death - birth + 1
            to_plot = np.zeros([l, 2])
            to_plot[:, 0] = np.asarray([k for k in range(birth, death+1)])
            for r in range(l):
                to_plot[r, 1] = height
            height += 1
            H += [to_plot]
    plt.figure()
    cols = ['b', 'r', 'g']
    k = 0
    LEG = []
    while k < len(features):
        LEG += [mpatches.patch(color=cols[k], label=f'{k}-Homology')]
        for j in range(len(features[k])):
            bar = features[k][j]
            plt.plot(bar[:, 0], bar[:, 1], color=cols[k])
        k += 1
    plt.legend(handles=LEG)
    plt.title(title)
    plt.savefig(f'{save_root} Barcodes.png', format="png")
    return features


idx = np.random.randint(len(X), size=30)
data = np.c_[X, y]
data = data[idx, :]
euc_dist = euclidean_dist(data)
D = np.amax(euc_dist)
VR_list = []
eps = 0.01*D
step = 0.01*D
count = 0

while eps < 0.55*D:
    ball_mat = epsilon_ball(euc_dist, eps, "Ball")
    VR = construct_VietorisRips(ball_mat, 2)
    VR_list += [VR]
    draw_simplexes(
        data, VR, f'PH for 30 random point from Trumps approval rating eps={eps}. ', "CircData/PH_", count)
    eps += step
    count += 1
