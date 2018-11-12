#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <pthread.h>
#define ROW 10
#define COLUMN 50 
#define NUM_THREAD 10 // width of river

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

pthread_mutex_t map_mutex ;	// thread mutex  
pthread_cond_t map_cond ;   // thread condition variable 
char map[ROW+10][COLUMN] ; 
int logs[ROW+10] ;
int speed[ROW+10] ; 
bool isRunning = true ;
bool isLose = false ; 
bool isWin = false ;


/* Determine a keyboard is hit or not. 
 * If yes, return 1. If not, return 0. */
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void *log_move( void *t ){
	
	int my_id = *((int*)&t); 

	while(isRunning){

		/*  Move the logs  */
		if(my_id%2) {  // move log from right to left
			logs[my_id] = logs[my_id]-1;
			if(logs[my_id]<0) logs[my_id] += (COLUMN - 1); 
		}else{		  // move log from left to right
			logs[my_id] = (logs[my_id]+1) % (COLUMN - 1);
		}

		pthread_mutex_lock(&map_mutex) ; 
		
		for(int j=0; j < COLUMN-1; ++j) map[my_id][j]=' ';			   //draw the river
		map[my_id][COLUMN-1] = 0 ;
		for(int i=0, j = logs[my_id]; i<15; ++i, ++j){				   //draw the logs
			map[my_id][j%(COLUMN-1)]='='; 
		}
		for(int j=0; j < COLUMN-1; ++j)	map[ROW][j] = map[0][j] = '|'; //draw the river bank
		
		/*  Check keyboard hits, to change frog's position or quit the game. */
		if( kbhit() ){
			char dir = getchar() ; 
			if( dir == 'w' || dir == 'W' )	frog.x-- ;  						// UP
			if((dir == 'a' || dir == 'A' ) && frog.y != 0)	frog.y-- ;		    // LEFT
			if((dir == 'd' || dir == 'D' ) && frog.y != COLUMN-2)	frog.y++ ;	// RIGHT
			if((dir == 's' || dir == 'S' ) && frog.x != ROW)	frog.x++ ;		// DOWN
			if( dir == 'q' || dir == 'Q' ) isRunning = false ;  				// quit
		}
	
		/*  Check game's status  */
		if( map[frog.x][frog.y] == ' ' || map[frog.x][frog.y] == 0 ){//check if frog jumps into water
			isRunning = false; 
			isLose = true; 
		}else if(frog.y >= COLUMN-2 || frog.y <= 0){  			 	 // check if frog is out of map
			isRunning = false;
			isLose = true;  	
		}else if(frog.x == 0){										 // frog has reached river bank
			isRunning = false;
			isWin = true; 
		} 

		/*  Print the map on the screen  */
		if(isRunning){
			if(frog.x == my_id && map[frog.x][frog.y] == '='){
				// frog move with the log
				if(frog.x % 2) frog.y--; 
				else frog.y++;
			}
			/* move the cursor to the upper-left corner
			 * of the screen and clear the screen */
			printf("\033[0;0H\033[2J");		
			usleep(1000) ;
			map[frog.x][frog.y] = '0' ;  
			for(int i = 0; i <= ROW; ++i)	puts(map[i]);
		}
	
		pthread_mutex_unlock(&map_mutex) ; 
		usleep(speed[my_id] * 5000) ;   //move the log
	}
	pthread_exit(NULL) ; 
}

int main( int argc, char *argv[] ){
	
	srand(time(0)) ; 
	pthread_t threads[NUM_THREAD];

	pthread_mutex_init(&map_mutex, NULL) ;
	pthread_cond_init(&map_cond , NULL) ;

	// Initialize the river map
	memset(map, 0, sizeof(map)) ;
	printf("\e[?25l") ; //hide the cursor
	int i, j; 
	for(i=1; i<ROW; ++i){	
		for(j=0; j<COLUMN-1; ++j) map[i][j] = ' ' ;  
		logs[i] = i ;
		speed[i] = rand()%20 + 10;  
	}
	for(j=0; j < COLUMN-1; ++j)	map[ROW][j] = map[0][j] = '|' ;
	// Initialize frog's starting position
	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	//map[frog.x][frog.y] = '0' ; 

	// Print the map into screen
	puts(map[ROW]);  
	
	/*  Create pthreads for wood move and frog control.  */
	for(i=1; i<NUM_THREAD; ++i){
		pthread_create(&threads[i], NULL, log_move, (void*)i) ;  
		usleep(200); 
	}
	for(i=1; i<NUM_THREAD; ++i){
		pthread_join(threads[i], NULL) ; 
	}

	/* move the cursor to the upper-left corner
	 * of the screen and clear the screen.    */
	printf("\033[0;0H\033[2J\033[?25h");
	usleep(1000) ;
	
	/*  Display the output for user: win, lose or quit.  */
	if(isLose){
		puts("You lose the game!!");
	}else if(isWin){
		puts("You win the game!!");
	}else {
		puts("You exit the game.");
	}

	/* clean up and exit */
	pthread_mutex_destroy(&map_mutex) ;	// destroy mutex
	pthread_cond_destroy(&map_cond) ;   // destroy condition 
	pthread_exit(NULL) ;			    // exit 
}
