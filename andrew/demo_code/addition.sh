#!/bin/bash

if [ $# -eq 1 ]
then
	echo "There is 1 number passed to the script"
	echo "Please input two numbers only"
fi

if [ $# -eq 2 ]
then
	echo "There are 2 numbers passed to the script"
	results=$(( $1 + $2))
	echo "The total of the $# number is $results"
fi

if [ $# -ge 3 ]
then
	echo "There are $# number passed to the script"
	echo "Please input two numbers only"
fi


