Running Dedalus on AWS
----------------------

Currently, dedalus can be run easily on single AWS instances. This means, you launch an instance and dedalus is preinstalled. All you need to do is transfer the python scripts that you wish to run. The example scripts from the dedalus codebase are also preinstalled.

First, you need an AWS account. Then sign into the console. Amazon's [tutorial](https://aws.amazon.com/getting-started/tutorials/launch-a-virtual-machine/)
will help you get most of the way with launching your first dedalus instance.


Once you are logged into the EC2 console, then click on "launch instance". The first choice is to pick an AMI. Under community AMIs search for 'dedalus'. You should see one AMI with the ID 'ami-361ba44e'. Select this AMI.

Next, select what instance type you would like to run on. The simplest example is to use the free default t2.micro.

Next, configure instance details. Again, the defaults can be used for the simplest example. If you prefer to use all the default settings you can click 'Review and Launch' instead of clicking through the remainder of the options for storage, tags, and security groups. 

If you choose not to use a default security group, make sure that the security group you use has a rule allowing SSH traffic. 

After you review your instance, click 'Launch'

If this is your first AWS instance, you'll be prompted to create an ssh key. Otherwise, choose which of your existings keys to use for this instance. 

Once the instance is running, find it's public IP address. From your command line you can ssh into this instance using:

``ssh -i /path/to/your_key.pem ubuntu@ip_address``

Please note that unlike the given tutorial which uses 'ec2-user@ip_address' for the dedalus AMI you must log in with 'ubuntu' as the username.

Once you're logged onto the instance there are example python scripts located in dedalus/src/dedalus/examples. If you need to transfer your own files to the instance you can do so with 

``scp -i /path/to/your_key.pem my_file ubuntu@ip_address:``

which will leave the file in your home directory on the instance, or, 

``scp -i /path/to/your_key.pem my_file ubuntu@ip_address:/path/to/desired/location``

which will leave the file in the desired location. 