	.file	"1.c"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	xorpd	%xmm0, %xmm0
	movl	$temp2.1581, %eax
	movl	$temp2.1581+67108864, %edx
	.p2align 4,,10
	.p2align 3
.L2:
	movapd	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	.L2
	xorl	%eax, %eax
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.local	temp2.1581
	.comm	temp2.1581,67108864,32
	.ident	"GCC: (Ubuntu/Linaro 4.7.3-1ubuntu1) 4.7.3"
	.section	.note.GNU-stack,"",@progbits
