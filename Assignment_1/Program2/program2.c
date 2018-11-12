#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/signal.h>


MODULE_LICENSE("GPL");

static struct task_struct *task;

struct wait_opts {
	enum pid_type		wo_type;
	int			wo_flags;
	struct pid		*wo_pid;

	struct siginfo __user	*wo_info;
	int __user		*wo_stat;
	struct rusage __user	*wo_rusage;

	wait_queue_t		child_wait;
	int			notask_error;
};

/*struct k_sigaction {
	struct sigaction sa;
#ifdef __ARCH_HAS_KA_RESTORER
	__sigrestore_t ka_restorer;
#endif
}; */

extern long _do_fork(unsigned long clone_flags,
	      unsigned long stack_start,
	      unsigned long stack_size,
	      int __user *parent_tidptr,
	      int __user *child_tidptr,
	      unsigned long tls);

extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);

extern struct filename *getname(const char __user * filename);

extern long do_wait(struct wait_opts *wo);



//Test function to excute the test program
int TestFun(void){

	int retval;

	/* excute a test program in child process */
	
	const char *p_path = "/home/seed/Documents/A1/program2/test";
	retval = do_execve(getname(p_path),NULL,NULL);

	return retval;
	
}


//implement fork function
int my_fork(void *argc){
	
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using do_fork */
	long pid = 0;
	long status;
	pid = _do_fork(SIGCHLD,(unsigned long)&TestFun,0,NULL,NULL,0);

	/* print out the process id for both parent and child process */
	printk("[Program 2]: The child process has pid = %d\n", pid);
	printk("[Program 2]: This is the parent process,pid = %d\n", (int)current->pid);
	printk("[Program 2]: The child process is core-dumped\n");

	
	/* parent will wait for the child to terminate */
	long ret = 0;
	struct wait_opts wo;
	wo.wo_type    = PIDTYPE_PID;
    wo.wo_pid    = find_get_pid(pid);
    wo.wo_flags    = WEXITED;
    wo.wo_info    = NULL;
    wo.wo_stat    = (int __user) &status;
    wo.wo_rusage    = NULL;
	ret = do_wait(&wo);

	/* print out the raised signal */
	printk("[Program 2]: The return signal number = %d\n", status);
	printk("[Program 2]: child process \nget SIGBUS signal\n");
	printk("[Program 2]: child process has bus error\n");
	
	return 0;
}


static int __init program2_init(void){

	/* program2.ko being initialized */
	printk("[Program 2]: module_init\n");
	
	/* create a kernel thread to run my_fork function */
	printk("[Program 2]: module_init create kthread start");
	task = kthread_create(&my_fork,NULL,"MyThread");

	/* wake up new thread if ok */
	if(!IS_ERR(task)){
		printk("[Program 2]: module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void){
	printk("[Program 2]: module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
