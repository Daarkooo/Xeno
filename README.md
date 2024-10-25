# Xeno 


```bash
docker-compose up -d



### Running the Server
To start the iperf server, run the following command in your terminal:

```bash
docker exec -it xeno-server-1 iperf -s


### check network configuration 

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'  xeno-server-1


# process 

### start containers
```bash
docker run -dit --name client1 --network my_custom_network --privileged ubuntu bash
docker run -dit --name client2 --network my_custom_network --privileged ubuntu bash
docker run -dit --name router --network my_custom_network --privileged ubuntu bash
 
# --privileged flag allows to use tc and other low-level network tools inside the container

### Install the necessary package (iproute2) in each container
```bash
docker exec client1 apt update && docker exec client1 apt install -y iproute2
docker exec client2 apt update && docker exec client2 apt install -y iproute2
docker exec router apt update && docker exec router apt install -y iproute2

# --This installs the iproute2 package, which includes the tc command.
