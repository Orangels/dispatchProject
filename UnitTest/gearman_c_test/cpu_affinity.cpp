//
// Created by Orangels on 2020-04-14.
//

#include <iostream>
#include <thread>
#include<stdlib.h>
#include<stdio.h>
#include<sys/types.h>
#include<sys/sysinfo.h>
#include<unistd.h>

#define __USE_GNU
#include<sched.h>
#include<ctype.h>
#include<string.h>
#include<pthread.h>

//g++ -o cpu_affinity cpu_affinity.cpp

using namespace std;

/* This method will create processes, then bind each to its own cpu. */
void do_cpu_stress(int num_of_process)
{
    int created_process = 0;
    /* We need a process for each cpu we have... */
    while ( created_process < num_of_process - 1 )
    {
        int mypid = fork();
        cout << "mypid " << mypid << endl;
        if (mypid == 0) /* Child process */
        {
            break;
        }
        else /* Only parent executes this */
        {
            /* Continue looping until we spawned enough processes! */ ;
            created_process++;
        }
    }
    /* NOTE: All processes execute code from here down! */
    cpu_set_t mask;
    /* CPU_ZERO initializes all the bits in the mask to zero. */
    CPU_ZERO( &mask );
    /* CPU_SET sets only the bit corresponding to cpu. */
    CPU_SET(created_process, &mask );
    /* sched_setaffinity returns 0 in success */
    if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 ){
        cout << "WARNING: Could not set CPU Affinity, continuing..." << endl;
    }
    else{
        cout << "Bind process #" << created_process << " to CPU #" << created_process << endl;
    }
    //do some cpu expensive operation
    int cnt = 100000000;
    while(cnt--){
        int cnt2 = 10000000;
        while(cnt2--){
        }
    }
}

int main(){
    int num_of_cpu = thread::hardware_concurrency();
    cout << "This PC has " << num_of_cpu << " cpu." << endl;
    do_cpu_stress(num_of_cpu);
}