import argparse
ap=argparse.ArgumentParser()
ap.add_argument("-n","--name",required=True,help="name of the user")
args=vars(ap.parse_args())
print("Hi {} There it's friendly to meet you !".format(args["name"]))
print(args)
#To invoke from command line call python sample_argparser.py -n 'PRASHANTH' or python sample_argparser.py --name "PRASHANTH"
#For help we can use python sample_argparser.py --help