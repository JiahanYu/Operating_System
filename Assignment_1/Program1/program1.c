#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]) {
    pid_t pid;
    int status;

    printf("Process start to fork\n");
    /* fork a child process */
    pid = fork();

    if(pid == -1){ /* error occurred */
        perror("fork");
        exit(1);
    }
    else{

        if(pid == 0){ /* child process */
            int i;
            char *arg[argc];

            sleep(3);
            printf("I'm the Child Process, my pid = %d\n", getpid());

            for(i=0;i<argc-1;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;

            printf("Child Process start to execute the program\n");
            /* execute test program */
            execve(arg[0],arg,NULL);

            raise(SIGCHLD);

            perror("execve");
            exit(EXIT_FAILURE);

        }

        else{ /* parent process */
            printf("I'm the Parent Process, my pid = %d\n", getpid());
            /* parent will wait for the child to terminate */
            waitpid(-1, &status, WUNTRACED);
            
            /* check child process's termination status */
            printf("Parent process receiving the SIGCHLD signal\n");
            if(WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)){
                printf("child process get SIGTERM signal\n");
                printf("child process is terminated by termination signal\n");
                printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
            }
            else if(WIFSTOPPED(status)){
                printf("child process get SIGSTOP signal\n");
                printf("child process stopped\n");
                printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
        }
    }

    return 0;
}
