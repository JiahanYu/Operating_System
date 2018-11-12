#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

int ptree[];

int myfork(int i, char *arg[]){
	pid_t p_pid;
	int p_status;
	
	/* fork a child process */
    p_pid = fork();

	if(p_pid == -1){ /* error occurred */
	    perror("fork");
	    exit(1);
	}
	else{

        if(p_pid == 0){ /* child process */

            /* execute test program */
            execve(arg[i],arg,NULL);
            
            if(arg[i]!=NULL){
            	myfork(i+1);
            }else{
            	return 0;
            }

	        perror("execve");
	        exit(EXIT_FAILURE);

	    }
	    else{ /* parent process */
			printf("The child process (pid=%d) of parent process(pid=%d)", get_pid(); current-> );
		}
	}

}


int main(int argc,char *argv[]){
    
    pid_t pid;
	int status;
	int i;
	char *arg[argc];
	int ptree[argc-1];


	for(i=0;i<argc-1;i++){
	    arg[i]=argv[i+1];
	}
	arg[argc-1]=NULL;

	myfork(0, *arg);

    return 0;
}
