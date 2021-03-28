import numpy as np

def predict(X,Theta):
	return X @ Theta
#Compute cost function
def computeCost(X,y,Theta):
    predicted = predict(X,Theta)
    sqr_error = (predicted - y)**2
    sum_error = np.sum(sqr_error)
    m = np.size(y)
    J = (1/(2*m))*sum_error
    return J

#computeCost_Vec
def computeCost_Vec(X,y,Theta):
    error = predict(X,Theta) - y
    m = np.size(y)
    J = (1/(2*m))*np.transpose(error)@error
    return J

def printProgressBar (iteration, total, suffix = ''):
    percent = ("{0:." + str(1) + "f}").format(100 * ((iteration+1) / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '=' * filledLength + '-' * (50 - filledLength)
    print('\rTraining: |%s| %s%%' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


#giá trị mặc định của alpha là 0.02, iter (số vòng lặp tối đa) là 5000
def GradientDescent(X,y,alpha=0.02,iter = 5000): 
    #Giá trị ban đầu của theta = 0
    theta = np.zeros(np.size(X,1)) #số lượng theta bằng số cột của X
    #array lưu lại các giá trị J trong quá trình lặp
    J_hist = np.zeros((iter,2)) # kích thước là iter*2, cột đầu chỉ là các số từ 1 đến iter để tiện cho việc plot. Kích thước được truyền vào qua một tupple
    #kích thước của training set
    m = np.size(y)
    #ma trận ngược (đảo hàng và cột) của X
    X_T = np.transpose(X)
    #biến tạm để kiểm tra tiến độ Gradient Descent
    pre_cost = computeCost(X,y,theta)
    for i in range(0,iter):
        printProgressBar(i,iter)
        #tính sai số (predict – y)
        error = predict(X,theta) - y
        #thực hiện gradient descent để thay đổi theta
        theta = theta - (alpha/m)*(X_T @ error)
        #tính J hiện tại
        cost = computeCost(X,y,theta)
        #so sánh với J của vòng lặp trước, so sánh 15 chữ số thập phân
        if np.round(cost,15) == np.round(pre_cost,15):
            #in ra vòng lặp hiện tại và J để dễ debug
            print('Reach optima at I = %d ; J = %.6f'%(i,cost))
            #thêm tất cả các index còn lại sau khi break
            J_hist[i:,0] = range(i,iter)
            #giá trị J sau khi break sẽ như cũ
            J_hist[i:,1] = cost
            #thoát vòng lặp
            break
        #cập nhật pre_cost
        pre_cost = cost
        #lưu lại index vòng lặp hiện tại
        J_hist[i, 0] = i
        #lưu lại J hiện tại
        J_hist[i, 1] = cost
    yield theta
    yield J_hist

def Normalize(X):
    #tạo copy của X (tham chiếu X) để không ảnh hưởng trực tiếp đến X (tham trị).
    n = np.copy(X)
    #x0 đầu tiên giả = 100
    n[0,0] = 100
    #tính std cho từng feature x
    s = np.std(n,0,dtype = np.float64)
    #tính mean cho từng feature x
    mu = np.mean(n,0)
    m = n- mu
    n = m/s
    #gán lại x0 = 1
    n[:,0] = 1
    yield n
    yield mu
    yield s

def Loadtxt(path):
    try:
        raw = np.loadtxt(path,delimiter = ',')
        X = np.zeros((np.size(raw,0),np.size(raw,1)))
        X[:,0] = 1
        X[:,1:] = raw[:,:-1]
        y = raw[:,-1]
        yield X
        yield y
    except:
        return 0
