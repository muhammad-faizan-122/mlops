# MLOps
Goal is to built and deploy the classifier.

## Run
- build and run docker.
```
docker build -t ml-classifier .
docker run --name classifier1 -p 8000:8000 ml-classifier
```
- **demo**
![alt text](test/demo1.png)

**Reach out**
```
http://localhost:8000/doc
```
![alt text](test/demo2.png)