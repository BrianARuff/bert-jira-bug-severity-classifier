import pandas as pd
import random

class DataGenerator:
    """
    Generates realistic bug descriptions using 5-tier severity scoring (1-5)
    based on Atlassian severity levels.
    """
    
    def __init__(self):
        # Building blocks for creating varied bug reports
        self.components = [
            'authentication service', 'payment gateway', 'user dashboard', 
            'reporting module', 'API endpoint', 'database connection',
            'file upload service', 'search functionality', 'email service',
            'mobile app', 'checkout process', 'admin panel', 'caching layer',
            'notification system', 'user profile management', 'order tracking system',
            'inventory management', 'session management', 'third-party integration',
            'data synchronization service', 'backup system', 'audit logging'
        ]
        
        self.user_actions = [
            'uploading a file larger than 50MB through the bulk upload feature',
            'attempting to log in after being idle for more than 8 hours',
            'navigating from the dashboard to the reports section using the sidebar menu',
            'trying to complete a payment transaction during peak traffic hours',
            'accessing their profile settings while having multiple tabs open',
            'submitting a form with special characters in the input fields',
            'using the search functionality with complex filter combinations',
            'downloading a report that contains more than 10,000 records',
            'switching between different user roles within the same session',
            'attempting to save changes while the internet connection is unstable'
        ]
        
        self.technical_contexts = [
            'when the server is under high load (>85% CPU usage)',
            'during the daily backup process that runs between 2-4 AM',
            'when the Redis cache is being cleared or updated',
            'after a recent deployment that included database schema changes',
            'when multiple users are accessing the same resource simultaneously',
            'during peak business hours when traffic exceeds normal capacity',
            'when the external API we depend on is experiencing rate limiting',
            'after a user\'s session has been idle for exactly 10 hours and 30 seconds',
            'when the database connection pool is near its maximum limit',
            'during maintenance windows when certain services are temporarily unavailable'
        ]
        
        # SEV-1: Critical incident with very high impact - service down for ALL customers
        self.sev1_impacts = [
            'causing complete system outage for all customers worldwide',
            'making the entire application inaccessible to all users',
            'preventing all customers from accessing any part of the service',
            'causing total service disruption affecting every user',
            'resulting in complete platform unavailability for all customers'
        ]
        
        # SEV-2: Major incident with significant impact - service down for SUBSET of customers
        self.sev2_impacts = [
            'affecting approximately 30% of our customer base',
            'making the service inaccessible for users in the European region',
            'preventing premium subscribers from accessing core features',
            'blocking all mobile users from completing transactions',
            'causing service outage for enterprise customers only'
        ]
        
        # SEV-3: Minor incident with low impact OR potential to become major
        self.sev3_impacts = [
            'creating minor inconvenience for customers during checkout',
            'causing partial loss of functionality for a small subset of users',
            'potentially escalating to major incident if not addressed quickly',
            'affecting non-critical features that some users rely on',
            'causing intermittent issues that could worsen over time'
        ]
        
        # SEV-4: Support request that's irritating but doesn't impact system function
        self.sev4_impacts = [
            'causing slower-than-average load times for certain pages',
            'creating minor usability issues that don\'t halt the product',
            'irritating customers but not preventing core functionality',
            'affecting user experience but allowing all functions to work',
            'causing performance degradation during specific operations'
        ]
        
        # SEV-5: Bugs that don't impact product usability
        self.sev5_impacts = [
            'showing the company logo in the wrong position on the footer',
            'displaying incorrect copyright year in the about section',
            'having a typo in help text that doesn\'t affect functionality',
            'showing misaligned text that doesn\'t obscure important information',
            'displaying wrong color for a decorative element'
        ]
        
        self.error_details = [
            'throwing a NullPointerException in the UserController.validateSession() method',
            'returning a 504 Gateway Timeout after exactly 30 seconds',
            'displaying a generic "Something went wrong" message with error code ERR_CONN_TIMEOUT',
            'causing a memory leak that gradually degrades system performance over 2-3 hours',
            'triggering a cascade failure that affects three downstream microservices',
            'generating thousands of error logs that fill up the disk space within minutes',
            'causing the load balancer to mark healthy servers as unhealthy',
            'resulting in database deadlocks that require manual intervention to resolve',
            'creating orphaned records in the database that break referential integrity',
            'causing SSL certificate validation failures for secure API calls'
        ]
        
        self.workarounds = [
            'Users can bypass this by clearing their browser cache and logging in again, but this loses their current work',
            'A temporary workaround is to access the feature through the mobile app, though functionality is limited',
            'Support staff can manually process these requests, but it takes 3x longer than the automated process',
            'Users can export data in smaller chunks (< 1000 records), but this defeats the purpose of bulk operations',
            'The feature works if accessed through incognito mode, suggesting a session storage issue',
            'Restarting the application server resolves the issue temporarily, but it recurs after 2-3 hours',
            'Using a different browser or device sometimes works, indicating a client-side caching problem',
            'The issue can be avoided by performing the action during off-peak hours (before 9 AM or after 6 PM)',
            'API calls work when made directly via Postman, suggesting the issue is in the frontend integration',
            'Rolling back to the previous version resolves the issue, but we lose important new features'
        ]

    def _get_detailed_template(self, severity):
        """Generates detailed templates based on 5-tier Atlassian severity levels."""
        
        templates = {
            1: [  # SEV-1: Critical - Service down for ALL customers
                """CRITICAL PRODUCTION OUTAGE - SEV-1 - ALL CUSTOMERS AFFECTED

Priority: P0 - IMMEDIATE RESPONSE REQUIRED
Impact: Complete service unavailability for all customers
Affected Users: 100% of customer base (approximately 50,000+ active users)

INCIDENT SUMMARY:
{technical_context}, the {component} has completely failed, {sev1_impact}. This is a SEV-1 critical incident requiring immediate escalation to senior engineering leadership.

TECHNICAL DETAILS:
The primary {component} {error_detail}, causing a complete cascade failure across our entire infrastructure. All customer-facing services are now returning 500 errors, and no users can access any functionality.

REPRODUCTION STEPS:
1. Any user attempts to access the application via web or mobile
2. User encounters immediate error page or infinite loading screen
3. All API endpoints return 500 Internal Server Error
4. Database connections are being refused
5. Load balancer health checks are failing across all regions

CURRENT STATUS:
- All monitoring dashboards showing red alerts
- Customer support receiving 200+ complaints per minute
- Social media mentions spiking negatively
- Revenue impact: $10,000+ per minute of downtime
- SLA breach imminent (we have 15 minutes left on our 99.9% uptime commitment)

BUSINESS IMPACT:
This outage is {sev1_impact}. We are violating our enterprise SLA agreements and face potential financial penalties. Customer trust and brand reputation are at severe risk.

ESCALATION PATH:
- VP of Engineering: NOTIFIED
- CTO: NOTIFIED  
- Customer Success: Preparing customer communications
- Legal: Standing by for SLA breach notifications

IMMEDIATE ACTIONS REQUIRED:
1. All hands on deck - cancel all meetings
2. Activate disaster recovery protocols
3. Prepare rollback to last known good deployment
4. Brief executive team every 15 minutes

WORKAROUND: 
{workaround}

War Room: Conference Room A / Zoom Bridge #emergency
Incident Commander: Senior Platform Engineer
Next Update: In 10 minutes""",

                """SECURITY BREACH - SEV-1 - SYSTEM COMPROMISE

Classification: CONFIDENTIAL - SECURITY INCIDENT
Severity: SEV-1 Critical
Discovery: Automated security monitoring alerts

SECURITY INCIDENT SUMMARY:
Our security monitoring detected that {technical_context}, an unauthorized actor has gained administrative access to the {component}. This represents a complete security compromise {sev1_impact}.

ATTACK VECTOR ANALYSIS:
The attacker exploited a vulnerability where {user_action} triggers the {component} to {error_detail}. This allows complete bypass of authentication and authorization controls, granting full administrative privileges.

CONFIRMED COMPROMISE:
1. Unauthorized access to customer PII database confirmed
2. Financial transaction logs accessed
3. Administrative functions executed by unknown entity
4. System audit logs show tampering attempts
5. Potential data exfiltration in progress

IMMEDIATE CONTAINMENT ACTIONS:
- Isolated affected {component} from network
- Revoked all active user sessions
- Disabled external API access
- Activated incident response team
- Notified legal and compliance teams

REGULATORY IMPACT:
This breach requires immediate notification to regulatory bodies within 72 hours. Potential GDPR fines could exceed $4M. All affected customers must be notified within 24 hours.

BUSINESS CONTINUITY:
{sev1_impact} until security clearance is obtained. Estimated recovery time: 6-12 hours minimum.

FORENSIC ANALYSIS REQUIRED:
1. Complete system imaging before cleanup
2. Log analysis for attack timeline
3. Data integrity verification
4. Customer impact assessment

Emergency Contact: CISO, Legal, Public Relations
Incident Response Team: ACTIVATED
Next Update: Every 30 minutes"""
            ],
            
            2: [  # SEV-2: Major - Service down for SUBSET of customers
                """MAJOR SERVICE DISRUPTION - SEV-2 - SUBSET OF CUSTOMERS AFFECTED

Priority: P1 - High
Impact: Significant service disruption for specific customer segments
Affected Users: Approximately 15,000 users (30% of customer base)

INCIDENT OVERVIEW:
{technical_context}, the {component} is experiencing critical failures that are {sev2_impact}. While not affecting all customers, this represents a significant business impact requiring immediate attention.

AFFECTED CUSTOMER SEGMENTS:
- Enterprise customers on the West Coast data center
- Premium subscribers using advanced features
- Mobile users accessing the application during peak hours
- International customers in the EU region

TECHNICAL ANALYSIS:
The {component} {error_detail} specifically when handling requests from the affected customer segments. Root cause analysis indicates:

- Database cluster in US-West region is experiencing high latency
- Load balancer health checks failing for 3 out of 8 servers
- Memory utilization on affected servers exceeds 95%
- Network connectivity issues between microservices

REPRODUCTION STEPS:
1. Log in as an enterprise customer from West Coast region
2. Attempt to {user_action}
3. System {error_detail} after 45-60 seconds
4. User session becomes unresponsive
5. Subsequent login attempts fail with timeout errors

BUSINESS IMPACT ASSESSMENT:
- Revenue at risk: $45,000 per hour from affected customers
- Enterprise SLA violations for 12 major accounts
- Customer satisfaction scores dropping rapidly
- Support ticket volume increased by 300%
- Potential contract renewals at risk

CUSTOMER COMMUNICATION:
- Status page updated with incident details
- Enterprise customers contacted directly
- Support team briefed on customer escalation procedures
- Executive team prepared for customer calls

MITIGATION PROGRESS:
1. Traffic rerouted to healthy data centers
2. Additional server capacity provisioned
3. Database optimization scripts running
4. Monitoring alerts configured for early detection

WORKAROUND:
{workaround}

Estimated Resolution: 2-4 hours
Incident Commander: Platform Engineering Manager
Next Update: In 1 hour""",

                """PAYMENT PROCESSING FAILURE - SEV-2 - FINANCIAL IMPACT

Priority: P1 - Critical Business Function Affected
Impact: Payment processing down for mobile customers
Revenue Impact: HIGH

FINANCIAL INCIDENT SUMMARY:
The {component} integration with our primary payment processor is failing for mobile customers, {sev2_impact}. This is causing direct revenue loss and customer frustration during our peak sales period.

AFFECTED TRANSACTIONS:
- Mobile app payments: 100% failure rate
- Subscription renewals: 85% failure rate  
- In-app purchases: 90% failure rate
- Refund processing: 75% failure rate

TECHNICAL ROOT CAUSE:
{technical_context}, the {component} {error_detail}. Investigation reveals:

- Payment gateway API returning HTTP 429 (rate limiting)
- Mobile SDK certificate expired at 2:15 PM
- Webhook delivery failures from payment processor
- Token refresh mechanism not working for mobile sessions

CUSTOMER JOURNEY IMPACT:
1. Customer adds items to cart (mobile app)
2. Proceeds to checkout successfully
3. Enters payment information
4. Clicks "Complete Purchase"
5. System {error_detail}
6. Payment appears to process but fails silently
7. Customer charged but order not created
8. Customer contacts support confused about payment status

FINANCIAL METRICS:
- Failed transactions: 1,247 in last 3 hours
- Average transaction value: $156
- Total revenue at risk: $194,532
- Customer service costs: $8,000 in overtime
- Potential chargeback fees: $15,000+

REGULATORY CONSIDERATIONS:
- PCI compliance review required
- Payment processor notification mandatory
- Transaction reconciliation needed
- Audit trail preservation required

IMMEDIATE ACTIONS:
1. Switch to backup payment processor for mobile
2. Manual verification of all affected transactions
3. Customer refund process initiated
4. Payment processor support escalation

WORKAROUND:
{workaround}

Finance Team: NOTIFIED
Payment Processor: ESCALATED
Customer Service: BRIEFED"""
            ],
            
            3: [  # SEV-3: Minor incident with low impact OR potential to escalate
                """MODERATE FUNCTIONALITY ISSUE - SEV-3 - POTENTIAL ESCALATION RISK

Priority: P2 - Medium
Impact: Partial functionality loss with escalation potential
Affected Users: Small subset experiencing workflow disruption

ISSUE SUMMARY:
Users are experiencing intermittent issues with the {component} that, while currently affecting a limited number of customers, {sev3_impact}. This requires monitoring to prevent escalation to a major incident.

CURRENT IMPACT ASSESSMENT:
- Approximately 8% of daily active users affected
- Issue occurs sporadically during peak usage hours
- Workaround available but requires manual intervention
- Potential to affect larger user base if left unresolved

TECHNICAL DETAILS:
{technical_context}, the {component} {error_detail}. This manifests as:

- Response times degraded from 2 seconds to 15+ seconds
- Intermittent timeouts during {user_action}
- Cache invalidation not working correctly
- Session persistence issues for affected users

REPRODUCTION STEPS:
1. Log in during peak hours (11 AM - 1 PM or 3 PM - 5 PM)
2. Navigate to {component} section
3. Attempt to {user_action}
4. Wait for system processing (takes 15+ seconds instead of normal 2-3 seconds)
5. Observe that {error_detail} occurs approximately 30% of the time
6. Retry the action - sometimes succeeds on second or third attempt

USER FEEDBACK:
"The {component} has been really slow lately. When I try to {user_action}, it either takes forever or just fails completely. I can usually get it to work if I try a few times, but it's really frustrating."

ESCALATION INDICATORS:
- Issue frequency increasing 15% daily
- More users reporting similar problems
- Performance degradation spreading to related features
- Error rates climbing during peak traffic

BUSINESS IMPACT:
- Minor decrease in user engagement metrics
- Slight increase in support ticket volume (+12%)
- Some users switching to competitor solutions
- Feature adoption rate declining for affected functionality

MONITORING AND ALERTS:
- Added specific monitoring for {component} performance
- Alert thresholds set for response time degradation
- User feedback tracking implemented
- Escalation path defined if issues worsen

PROPOSED RESOLUTION:
1. Optimize database queries in {component}
2. Implement better caching strategy
3. Add circuit breaker pattern for resilience
4. Scale infrastructure proactively

WORKAROUND:
{workaround}

Development Team: ASSIGNED
Product Owner: INFORMED
Timeline: 1-2 weeks for permanent fix""",

                """INTEGRATION INSTABILITY - SEV-3 - THIRD PARTY DEPENDENCY

Component: {component}
Priority: P2 - Monitor for Escalation
Impact: Intermittent failures with potential growth

INTEGRATION ISSUE OVERVIEW:
The {component} is experiencing intermittent connectivity issues with our third-party service provider. While currently manageable, {sev3_impact} if the provider's service degrades further.

THIRD PARTY SERVICE STATUS:
- Provider reporting "investigating performance issues"
- Their service availability: 94% (down from 99.8%)
- API response times increased 300%
- Webhook delivery delays: 5-15 minutes

AFFECTED WORKFLOWS:
When users {user_action}, the integration sometimes {error_detail}. This affects:

- Data synchronization between systems
- Real-time notifications to users
- Automated workflow triggers
- Reporting accuracy and timeliness

FAILURE PATTERNS:
- Failures occur in clusters during specific time windows
- Success rate varies by geographic region
- Retry mechanisms partially effective
- Manual intervention required for 15% of failed operations

DETAILED REPRODUCTION:
1. User initiates {user_action} that requires third-party integration
2. System makes API call to external service
3. {technical_context}, the external service {error_detail}
4. Our system waits for 30-second timeout
5. Retry logic attempts call 2 more times
6. After 90 seconds total, operation fails
7. User sees error message suggesting they try again later

BUSINESS CONTINUITY ASSESSMENT:
- Core functionality remains available
- Users can complete most tasks without this integration
- Data consistency maintained through eventual sync
- Revenue impact minimal but growing

MITIGATION STRATEGIES:
1. Implemented exponential backoff retry logic
2. Added circuit breaker to prevent cascade failures  
3. Created manual override process for critical operations
4. Negotiating SLA improvements with third-party provider

ESCALATION TRIGGERS:
- Failure rate exceeds 25%
- Provider downtime exceeds 2 hours
- Customer complaints increase significantly
- Core business processes become affected

WORKAROUND:
{workaround}

Vendor Management: ENGAGED
Architecture Review: SCHEDULED"""
            ],
            
            4: [  # SEV-4: Support request that's irritating but doesn't impact system function
                """USABILITY ISSUE - SEV-4 - PERFORMANCE DEGRADATION

Component: {component}
Priority: P3 - Low
Impact: User experience degradation without functional impact

PERFORMANCE ISSUE SUMMARY:
Users are reporting that the {component} is experiencing slower-than-average load times, particularly {technical_context}. While this doesn't prevent users from completing their tasks, {sev4_impact} and is generating support complaints.

USER EXPERIENCE IMPACT:
- Page load times increased from 3 seconds to 8-12 seconds
- Users experiencing frustration with perceived sluggishness
- Some users abandoning actions due to perceived unresponsiveness
- Overall user satisfaction scores decreased by 0.3 points

TECHNICAL ANALYSIS:
The performance degradation occurs when users {user_action}. Investigation shows:

- Database queries taking 2-3x longer than baseline
- Frontend JavaScript execution time increased
- Network requests showing higher latency
- Caching efficiency reduced by 15%

REPRODUCTION STEPS:
1. Log in to the application during business hours
2. Navigate to the {component} section
3. Attempt to {user_action}
4. Observe loading spinner for 8-12 seconds (normal: 3 seconds)
5. Page eventually loads correctly with all functionality intact
6. All features work as expected, just slower

USER FEEDBACK SAMPLES:
"Everything still works, but the {component} has gotten really slow lately. It's not broken, just annoying."

"The {component} takes forever to load now. I can still do my work, but it's frustrating having to wait so long."

FUNCTIONAL VERIFICATION:
- All core features working correctly
- Data integrity maintained
- No errors or failures occurring
- All user actions complete successfully (just slowly)

BUSINESS IMPACT ASSESSMENT:
- No revenue loss or functional blocking
- Minor decrease in user satisfaction metrics
- Slight increase in support inquiries (+8%)
- No impact on business operations or SLAs

OPTIMIZATION OPPORTUNITIES:
1. Database query optimization for {component}
2. Frontend code splitting and lazy loading
3. CDN configuration improvements
4. Caching strategy enhancement

WORKAROUND:
{workaround}

Performance Team: REVIEWING
Timeline: Next sprint planning cycle
Priority: Nice-to-have improvement""",

                """WORKFLOW INEFFICIENCY - SEV-4 - PROCESS IMPROVEMENT

Area: {component} 
Type: User Experience Enhancement
Priority: P3 - Process Optimization

WORKFLOW ANALYSIS:
The current {component} workflow is functional but inefficient, requiring users to {user_action} through multiple steps that could be streamlined. While users can complete their tasks, {sev4_impact} and impacts productivity.

CURRENT PROCESS FLOW:
1. User accesses {component} from main navigation
2. System loads interface (takes 5-8 seconds)
3. User must click through 4 different screens to {user_action}
4. Each screen transition requires 3-4 seconds loading
5. Form submission requires manual confirmation at each step
6. Total time to complete: 3-5 minutes (industry standard: 1-2 minutes)

USER FRUSTRATION POINTS:
- Too many confirmation dialogs for routine actions
- Repetitive data entry across multiple screens
- No bulk operations available for common tasks
- Progress is lost if user navigates away accidentally

PRODUCTIVITY IMPACT:
- Operations team spends 40% more time on routine tasks
- Daily workflow completion delayed by average 30 minutes
- Staff requesting training on "faster ways" to use system
- Some users developing unofficial workarounds

COMPARATIVE ANALYSIS:
{technical_context}, users report that the previous system allowed {user_action} in 2-3 clicks. The current system requires 8-10 clicks for the same outcome.

USER FEEDBACK:
"The new {component} works fine, but it takes so much longer to do simple things. I miss being able to {user_action} quickly. Now I have to click through so many screens."

ENHANCEMENT OPPORTUNITIES:
1. Combine multiple screens into single-page workflow
2. Add bulk action capabilities
3. Implement smart defaults and auto-completion
4. Reduce unnecessary confirmation steps
5. Add keyboard shortcuts for power users

BUSINESS JUSTIFICATION:
- Improved user satisfaction scores
- Increased productivity for operations team
- Reduced training time for new users
- Better competitive positioning

PROPOSED SOLUTION:
{workaround}

UX Team: CONSULTING
Product Manager: REVIEWING
Development Effort: 2-3 sprints"""
            ],
            
            5: [  # SEV-5: Bugs that don't impact product usability
                """COSMETIC ISSUE - SEV-5 - VISUAL INCONSISTENCY

Component: {component}
Type: UI/Visual Polish
Priority: P4 - Cosmetic Only

VISUAL ISSUE DESCRIPTION:
There is a minor visual inconsistency in the {component} where {sev5_impact}. This is purely cosmetic and has no impact on functionality or user ability to complete tasks.

SPECIFIC DETAILS:
- Company logo appears 3 pixels lower than design specifications
- Footer copyright text shows "2023" instead of current year "2024"
- Navigation menu item text color is #2D3748 instead of brand color #1A202C
- Button border radius is 4px instead of standard 6px used elsewhere

DISCOVERY METHOD:
This issue was identified during a routine design system audit. No users have reported this inconsistency, and it doesn't interfere with any user workflows or system functionality.

IMPACT ASSESSMENT:
- Zero functional impact on user experience
- No effect on task completion or workflow
- Brand consistency slightly affected
- Only noticeable when specifically looking for design inconsistencies

BROWSER COMPATIBILITY:
- Issue appears consistently across all supported browsers
- Mobile and desktop versions both affected
- Screen reader functionality unaffected
- Keyboard navigation works correctly

TECHNICAL REQUIREMENTS:
Simple CSS update required:
css
.component-logo {{
  margin-top: 12px; /* currently 15px */
}}
.footer-copyright {{
  content: "2024"; /* currently "2023" */
}}
.nav-item {{
  color: #1A202C; /* currently #2D3748 */
}}
.button-primary {{
  border-radius: 6px; /* currently 4px */
}}
BUSINESS PRIORITY:
- No customer impact or complaints
- No revenue or operational effect
- Design team perfectionism item
- Can be addressed during next maintenance cycle

EFFORT ESTIMATE:
- Developer time: 15 minutes
- QA testing: 30 minutes
- Design review: 15 minutes
- Total effort: 1 hour

RESOLUTION PLAN:
Include in next routine maintenance deployment alongside other minor fixes. No urgency required.""",

                """TEXT CORRECTION - SEV-5 - MINOR CONTENT ISSUE

Location: {component} Help Section
Type: Content/Documentation
Priority: P4 - Content Polish

CONTENT ISSUE SUMMARY:
The help text in the {component} contains a minor typo where "recieve" is spelled incorrectly instead of "receive". This is {sev5_impact} that doesn't affect user understanding or functionality.

EXACT CONTENT:
Current text: "You will recieve a confirmation email within 5 minutes."
Correct text: "You will receive a confirmation email within 5 minutes."

DISCOVERY:
- Identified during routine content audit
- No users have reported this typo
- Spell-check tools flagged during documentation review
- Similar typos found in 2 other help sections

FUNCTIONAL IMPACT:
- No impact on user task completion
- Message meaning clearly understood despite typo
- Email functionality works correctly
- User guidance remains effective

RELATED FINDINGS:
During the content audit, we also found:
- One instance of inconsistent capitalization in button labels
- Help tooltip showing placeholder text "[TBD]" in development environment
- One broken internal help link (404 error)

USER EXPERIENCE:
- Professional appearance slightly diminished
- No confusion about intended meaning
- Users successfully complete associated workflows
- Customer support has not received any related inquiries

CONTENT MANAGEMENT:
- Text stored in content management system
- Requires content editor access to update
- Change will be reflected immediately upon save
- No deployment or technical release required

QUALITY ASSURANCE:
1. Update content in CMS
2. Verify text displays correctly in all environments
3. Check for any other instances of the same typo
4. Update content style guide if needed

EFFORT REQUIRED:
- Content update: 5 minutes
- Verification: 10 minutes  
- Documentation: 5 minutes
- Total time: 20 minutes

This can be grouped with other minor content updates for the next content release cycle."""
            ]
        }
        
        return random.choice(templates[severity])

    def generate_bug(self, severity):
        """Generates a single detailed bug description."""
        template = self._get_detailed_template(severity)
        
        # Select appropriate impact based on severity
        if severity == 1:
            impact = random.choice(self.sev1_impacts)
        elif severity == 2:
            impact = random.choice(self.sev2_impacts)
        elif severity == 3:
            impact = random.choice(self.sev3_impacts)
        elif severity == 4:
            impact = random.choice(self.sev4_impacts)
        else:  # severity == 5
            impact = random.choice(self.sev5_impacts)
        
        bug = template.format(
            component=random.choice(self.components),
            user_action=random.choice(self.user_actions),
            technical_context=random.choice(self.technical_contexts),
            error_detail=random.choice(self.error_details),
            workaround=random.choice(self.workarounds),
            sev1_impact=random.choice(self.sev1_impacts),
            sev2_impact=random.choice(self.sev2_impacts),
            sev3_impact=random.choice(self.sev3_impacts),
            sev4_impact=random.choice(self.sev4_impacts),
            sev5_impact=random.choice(self.sev5_impacts)
        )
        return bug

    def generate_dataset(self, bugs_per_severity=500):
        data = []
        for severity in [1, 2, 3, 4, 5]:  # Now includes all 5 severity levels
            print(f"Generating {bugs_per_severity} severity {severity} bugs...")
            for _ in range(bugs_per_severity):
                data.append({
                    'description': self.generate_bug(severity),
                    'severity': severity
                })
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        return df

    def save_dataset(self, df, filename='dataset.xlsx'):
        df.to_excel(filename, index=False)
        print(f"Saved {len(df)} bugs to {filename}")
        print(f"Severity distribution:\n{df['severity'].value_counts().sort_index()}")


# This part remains the same
if __name__ == "__main__":
    # Create generator instance
    generator = DataGenerator() # Use the new generator
    
    # Generate dataset
    # Using a larger number to create more variety
    df = generator.generate_dataset(bugs_per_severity=3_000) # total_bugs = bugs_per_severity * 5 (1 for each severity)
    
    # Save to Excel
    generator.save_dataset(df, 'dataset.xlsx')
    
    # Also save to CSV for easier reading
    df.to_csv('dataset.csv', index=False)
    print(f"Also saved to dataset.csv")