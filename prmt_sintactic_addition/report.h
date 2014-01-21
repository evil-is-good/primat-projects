#ifndef CLASS_REPORT
#define CLASS_REPORT

#include <iostream>
#include <string>

namespace prmt
{
class Report
{
    public:
        Report ()
        {
            set_in_default ();
        };

        void operator= (Report incoming_report)
        {
            set_in_default ();

            who_what = incoming_report.who_what;
            result   = incoming_report.result;
            ther_is_a_message = incoming_report.ther_is_a_message;
            message = incoming_report.message;

            show_report();
        };

        void show_report ()
        {
            if (result _is success) then
                std::cout << who_what << " is success" << std::endl;
            else if (result _is not success) then
                std::cout << who_what << " is failure" << std::endl;

            if (ther_is_a_message) then
                std::cout << message << std::endl;
        };

        void set_in_default()
        {
            result   = false;
            who_what = "I do not know ";
            ther_is_a_message = false;
            message = "empty";
        };


        bool        result;
        std::string who_what;
        bool        ther_is_a_message;
        std::string message;

        static const bool success = true;

};
};

// Report::Report () 
// {
//     set_in_default ();
// };
// 
// void Report::operator= (Report incoming_report)
// {
//     set_in_default ();
// 
//     who_what = incoming_report.who_what;
//     result   = incoming_report.result;
//     ther_is_a_message = incoming_report.ther_is_a_message;
//     message = incoming_report.message;
// 
//     show_report();
// };
// 
// void Report::show_report ()
// {
//     if (result _is success) then
//         std::cout << who_what << " is success" << std::endl;
//     else if (result _is not success) then
//         std::cout << who_what << " is failure" << std::endl;
// 
//     if (ther_is_a_message) then
//         std::cout << message << std::endl;
// };
// 
// void Report::set_in_default()
// {
//     result   = false;
//     who_what = "I do not know ";
//     ther_is_a_message = false;
//     message = "empty";
// };

prmt::Report _report;

#define REPORT_USING

#ifdef REPORT_USING

#define REPORT _report = 

#define REPORT_USE(x) x

//_report .set_in_default ();
#define _return(report) _report .set_in_default (); \
                        report.who_what = __PRETTY_FUNCTION__;\
                        return report; 

#define _return_report _report .set_in_default (); \
                       prmt::Report __report;\
                       __report.who_what = __PRETTY_FUNCTION__;\
                       __report.result   = true;\
                       return __report;

#define _return_report_true _report .set_in_default (); \
                            prmt::Report __report;\
                            __report.who_what = __PRETTY_FUNCTION__;\
                            __report.result   = true;\
                            return __report;

#define _return_report_false _report .set_in_default (); \
                             prmt::Report __report;\
                             __report.who_what = __PRETTY_FUNCTION__;\
                             __report.result   = false;\
                             return __report;

#define OPERATOR_REPORT public:\
                        operator prmt::Report ()\
                        {\
                            prmt::Report report;\
                            report.who_what = __PRETTY_FUNCTION__;\
                            report.result   = true;\
                            return report;\
                        };\

#define OPERATOR_REPORT_SPECIAL public:\
                        operator prmt::Report ()\
                        {\
                            prmt::Report report;\
                            report.who_what = __PRETTY_FUNCTION__;\
                            report.result   = true;\
                            additional_reports_values (report);\
                            return report;\
                        };\
                        void additional_reports_values (Report);

#else

#define REPORT 

#define REPORT_USE(x)

//_report .set_in_default ();
#define _return(report) report.who_what = __PRETTY_FUNCTION__;\
                        return report; 

#define _return_report prmt::Report __report;\
                       __report.who_what = __PRETTY_FUNCTION__;\
                       __report.result   = true;\
                       return __report;

#define _return_report_true prmt::Report __report;\
                            __report.who_what = __PRETTY_FUNCTION__;\
                            __report.result   = true;\
                            return __report;

#define _return_report_false prmt::Report __report;\
                             __report.who_what = __PRETTY_FUNCTION__;\
                             __report.result   = false;\
                             return __report;

#define OPERATOR_REPORT public:\
                        operator prmt::Report ()\
                        {\
                            prmt::Report report;\
                            report.who_what = __PRETTY_FUNCTION__;\
                            report.result   = true;\
                            return report;\
                        };\

#define OPERATOR_REPORT_SPECIAL public:\
                        operator prmt::Report ()\
                        {\
                            prmt::Report report;\
                            report.who_what = __PRETTY_FUNCTION__;\
                            report.result   = true;\
                            additional_reports_values (report);\
                            return report;\
                        };\
                        void additional_reports_values (Report);
#endif


#endif
