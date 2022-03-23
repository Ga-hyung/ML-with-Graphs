Node2Vec 모델은 Node2Vec 파일 안에 코드가 있습니다.
	Wikipedia 에서 다운 받은 POS.mat 데이터와, Wikipedia 데이터 preprocessing을 위한 .py 데이터가 있고, 
	src 안에 main.py 와 node2vec.py그리고 예측에 사용한 predict_netmf.py 파일이 있습니다.
	predict_netmf.py 파일은 NetMF 공식 github에서 사용한 코드와 동일합니다.

	Length of Walk 파라미터를 달리하여 embedding한 값들은 .npz 파일로 embedding 폴더 안에 들어있습니다.

NetMF 모델은 NetMF 파일 안에 코드가 있습니다. 
	NetMF 자체를 embedding한 .npz 파일과, netmf.py 그리고 predict.py 파일이 존재합니다.

유사도를 계산하기 위해 사용한 코드는 similarty.py 안에 있습니다.
	